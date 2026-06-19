#!/usr/bin/env python3
"""
autoprime harness — the immutable judge.

This is the autoprime analog of autoresearch's `evaluate_bpb`: it is READ-ONLY
for the agent. The agent owns everything under `solve/`; this file is the one
fixed thing in the loop.

What it does, every run:
  1. detect the machine (arch, chip, cores, gmp prefix) and expose it via env
  2. BUILD the workspace by running `solve/build.sh` (untimed)
  3. RUN the produced `solve/run` for a fixed time budget (timed, hard-killed)
  4. VERIFY every claimed prime independently, rejecting Mersenne numbers and
     anything that isn't actually prime. This is untimed and rigorous: the test
     is BPSW (deterministic for all n < 2**64), run in C via gmpy2 and fanned
     out across cores so hundreds of millions of claims verify in well under a
     minute instead of ~12. Huge `largest` claims (n >= 2**64) keep the exact
     sympy path. The capture file is read in blocks, never slurped whole, so a
     multi-GB run stays in bounds. Speed here never comes at correctness's cost.
  5. SCORE and print a grep-able summary block. `count` is uncapped: the score
     is simply how many distinct primes were verified, so it can climb forever.

The judge is deliberately strategy-agnostic. It does not care HOW a number was
produced — pure C, a Metal GPU kernel, an ML predictor's guess, whatever. It
only checks that each claimed number is genuinely prime, non-Mersenne, and was
emitted within the budget. A bogus guess simply fails verification and scores
nothing. That neutrality is what makes the wild approaches safe to allow.

Protocol: the runner prints lines of the form `PRIME <decimal>` to stdout,
flushed as found. Anything else on stdout (and all of stderr) is ignored by the
scorer but passed through for debugging.

Usage:
  python3 harness.py --task largest          # biggest non-Mersenne prime in 60s
  python3 harness.py --task count            # most distinct primes in 60s
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import multiprocessing as mp

import numpy as np
import sympy

# gmpy2's primality test is the same BPSW used by sympy (deterministic for every
# n < 2**64), but runs in C — far faster across the hundreds of millions of
# small claims a `count` run produces. Optional: fall back to sympy if absent.
try:
    import gmpy2
    _HAVE_GMPY2 = True
except ImportError:                                  # pragma: no cover
    _HAVE_GMPY2 = False

# Python 3.11+ defaults to rejecting decimal-to-int conversions above a few
# thousand digits. Autoprime's largest task intentionally goes far beyond that;
# let SymPy, not the parser guardrail, decide how large a verified prime can be.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

# --- fixed constants (do not depend on the agent) ---------------------------
TIME_BUDGET = 60          # wall-clock run budget in seconds
GRACE_SECONDS = 5         # extra slack before the harness hard-kills the run
POLL_INTERVAL = 0.05      # how often we check the run's deadline

# Memory-safety valve only — NOT a scoring ceiling. The judge must remember
# every DISTINCT claim to reject duplicates, so an unbounded solver could
# otherwise exhaust RAM. This guard sits far above any presently reachable
# count; the `count` score is open-ended below it. Raise it if your hardware
# allows. (This replaces the old hard 5M cap: `count` is now a forever climb.)
CLAIM_SAFETY_LIMIT = 200_000_000

PRIME_RE = re.compile(rb"^\s*PRIME\s+([0-9]+)\s*$")
# Multiline variant for block-scanning the capture file in one C-level pass.
_CLAIM_RE = re.compile(rb"(?m)^[ \t]*PRIME[ \t]+([0-9]+)[ \t]*\r?$")
# Claims with at most this many digits fit in a signed 64-bit int (< 10**18 <
# 2**63), so they ride the vectorized numpy/gmpy2 fast path. Longer claims (the
# `largest` task's thousand-digit primes) take the exact big-integer path.
_INT64_DIGITS = 18


def _sysctl(key):
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _brew_prefix(pkg):
    try:
        return subprocess.check_output(
            ["brew", "--prefix", pkg], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return os.environ.get("AUTOPRIME_GMP_PREFIX", "")


def detect_specs():
    """What hardware does the agent have to play with?"""
    return {
        "arch": os.uname().machine,
        "chip": _sysctl("machdep.cpu.brand_string") or "unknown",
        "cores": os.cpu_count() or 1,
        "gmp_prefix": _brew_prefix("gmp"),
    }


def build(workspace, specs, task, time_budget):
    """Run the workspace's own build.sh. It must produce an executable ./run."""
    build_sh = os.path.join(workspace, "build.sh")
    if not os.path.isfile(build_sh):
        raise RuntimeError(f"no build.sh found in {workspace!r}")

    env = os.environ.copy()
    env["AUTOPRIME_ARCH"] = specs["arch"]
    env["AUTOPRIME_CORES"] = str(specs["cores"])
    env["AUTOPRIME_GMP_PREFIX"] = specs["gmp_prefix"]
    env["AUTOPRIME_TASK"] = task
    env["AUTOPRIME_TIME"] = str(time_budget)

    print(f"[harness] building: bash build.sh  (cwd={workspace})", flush=True)
    r = subprocess.run(["bash", "build.sh"], cwd=workspace, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"build.sh failed with exit code {r.returncode}")

    run_path = os.path.join(workspace, "run")
    if not (os.path.isfile(run_path) and os.access(run_path, os.X_OK)):
        raise RuntimeError("build.sh did not produce an executable ./run")
    return os.path.abspath(run_path)


def run_solver(run_path, workspace, task, time_budget):
    """Run ./run for the budget, capturing its stdout (the PRIME channel).

    stdout goes to a unique temp file so a hard-kill never loses already-flushed
    lines and parallel harness invocations do not clobber each other. Only bytes
    present at the official deadline count; the grace period is for cleanup, not
    extra scoring time. stderr is inherited (shows up in run.log) for debugging.
    The process gets its own session so we can kill the whole tree at once.
    """
    workspace = os.path.abspath(workspace)
    fd, out_path = tempfile.mkstemp(prefix=".run.", suffix=".out", dir=workspace)
    start = time.monotonic()
    cutoff_size = None
    killed = False
    with os.fdopen(fd, "w+b") as out:
        proc = subprocess.Popen(
            [run_path, "--task", task, "--time", str(time_budget)],
            cwd=workspace,
            stdout=out,
            stderr=None,
            start_new_session=True,
        )
        deadline = start + time_budget
        kill_deadline = deadline + GRACE_SECONDS
        while True:
            if proc.poll() is not None:
                if cutoff_size is None:
                    out.flush()
                    cutoff_size = os.fstat(out.fileno()).st_size
                break
            now = time.monotonic()
            if cutoff_size is None and now >= deadline:
                out.flush()
                cutoff_size = os.fstat(out.fileno()).st_size
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if now >= kill_deadline:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                killed = True
                break
            time.sleep(POLL_INTERVAL)
        out.flush()
    if cutoff_size is None:                       # never hit the deadline path
        cutoff_size = os.path.getsize(out_path)
    elapsed = time.monotonic() - start
    if killed:
        print(
            f"[harness] run exceeded {time_budget}+{GRACE_SECONDS}s — hard-killed",
            file=sys.stderr,
            flush=True,
        )
    # Hand back the capture file and the official byte cutoff. Scoring streams it
    # line by line so even a multi-GB run never has to live in memory at once.
    # The caller deletes the file once it has finished scoring.
    return out_path, cutoff_size, elapsed


def is_mersenne(n):
    """True if n == 2^k - 1 (i.e. n+1 is a power of two)."""
    m = n + 1
    return m > 1 and (m & (m - 1)) == 0


def is_genuine_prime(n):
    """Ground truth for one claim: a real, non-Mersenne prime.

    Both backends are BPSW, which is deterministic (no composite passes) for
    every n < 2**64 — covering any prime a `count` solver can enumerate, so a
    verified claim is certain, not merely probable. gmpy2 runs that same test in
    C; for n >= 2**64 (the `largest` task's huge primes) we defer to sympy, the
    original verifier, so nothing about that path's rigor changes. Still untimed:
    the judge takes as long as it needs and never trades correctness for speed.
    """
    if n <= 1 or is_mersenne(n):
        return False
    if _HAVE_GMPY2 and n < (1 << 64):
        return gmpy2.is_prime(int(n)) != 0
    return bool(sympy.isprime(n))


def _verify_small_chunk(arr):
    """Worker: count genuine primes in a non-Mersenne int64 array.

    Inputs are already filtered to n > 1 and non-Mersenne, and are < 2**64, so a
    plain BPSW test settles each one deterministically. Returns (count, max) so
    the parent can total the verified count and recover the largest prime.
    """
    primality = gmpy2.is_prime if _HAVE_GMPY2 else sympy.isprime
    cnt = 0
    best = 0
    for x in arr.tolist():                 # python ints iterate faster than np scalars
        if primality(x):
            cnt += 1
            if x > best:
                best = x
    return cnt, best


def _verify_small(cand, workers):
    """Verify a Mersenne-filtered int64 array of candidates, fanning out on cores.

    Primality is embarrassingly parallel and dominates a big `count` verify, so
    we split the candidates across a process pool. Small inputs (and the
    `largest` task, which has almost none here) skip the pool entirely.
    """
    n = int(cand.size)
    if n == 0:
        return 0, None
    if n < 1_000_000 or workers <= 1:
        cnt, best = _verify_small_chunk(cand)
        return cnt, (best or None)
    total = 0
    best = 0
    chunks = np.array_split(cand, workers * 4)
    with mp.Pool(workers) as pool:
        for cnt, b in pool.imap_unordered(_verify_small_chunk, chunks):
            total += cnt
            if b > best:
                best = b
    return total, (best or None)


def _read_claims(out_path, cutoff_size):
    """Stream the capture file up to the deadline cutoff and extract every claim.

    Reads in large blocks and pulls all `PRIME <n>` matches per block with one
    C-level regex pass — no Python-level loop over hundreds of millions of lines.
    Only bytes before cutoff_size count; a line straddling the cutoff (or an
    unterminated final line) is left in the carry buffer and dropped, exactly as
    the streaming scorer required. Claims that fit in int64 are returned as an
    ndarray for vectorized handling; longer ones (huge `largest` primes) come
    back as a separate list of Python ints.
    """
    small_chunks = []
    big = []
    remaining = cutoff_size
    carry = b""
    BLOCK = 1 << 26                                   # 64 MiB
    with open(out_path, "rb") as f:
        while remaining > 0:
            block = f.read(BLOCK if BLOCK < remaining else remaining)
            if not block:
                break
            remaining -= len(block)
            data = carry + block
            nl = data.rfind(b"\n")                    # process only complete lines
            if nl < 0:
                carry = data
                continue
            body, carry = data[:nl + 1], data[nl + 1:]
            nums = _CLAIM_RE.findall(body)
            if not nums:
                continue
            small = [x for x in nums if len(x) <= _INT64_DIGITS]
            if small:
                small_chunks.append(np.array(small, dtype=np.int64))
            if len(small) != len(nums):
                big.extend(int(x) for x in nums if len(x) > _INT64_DIGITS)
    small = (np.concatenate(small_chunks) if small_chunks
             else np.empty(0, dtype=np.int64))
    return small, big


def score_run(out_path, cutoff_size):
    """Read the capture file up to the deadline cutoff and score it.

    Dedup distinct `PRIME <n>` claims, reject Mersenne, verify the rest, and
    track how many verified and the maximum. The common case — a `count` run of
    hundreds of millions of small primes — is handled with numpy (block parse,
    sort-dedup, vectorized Mersenne reject) and a parallel BPSW sweep, turning a
    ~12-minute verify into well under a minute without weakening any check. Huge
    `largest` claims keep the exact per-number path. CLAIM_SAFETY_LIMIT still
    caps how many distinct claims we score, a far-off memory guard.
    """
    small, big = _read_claims(out_path, cutoff_size)

    small_distinct = np.unique(small) if small.size else small   # sorted + deduped
    big_distinct = list(dict.fromkeys(big))                      # order-preserving

    # Memory guard: score at most CLAIM_SAFETY_LIMIT distinct claims (small first).
    if small_distinct.size >= CLAIM_SAFETY_LIMIT:
        small_distinct = small_distinct[:CLAIM_SAFETY_LIMIT]
        big_distinct = []
    else:
        big_distinct = big_distinct[:CLAIM_SAFETY_LIMIT - int(small_distinct.size)]
    num_claimed = int(small_distinct.size) + len(big_distinct)

    # Small claims: drop n <= 1 and Mersenne (n+1 a power of two) vectorized, then
    # verify what's left in parallel.
    if small_distinct.size:
        d = small_distinct
        cand = d[(d > 1) & (((d + 1) & d) != 0)]
    else:
        cand = small_distinct
    workers = os.cpu_count() or 1
    num_verified, best = _verify_small(cand, workers)

    # Huge claims (e.g. the `largest` task): exact, untimed, one at a time.
    for n in big_distinct:
        if is_genuine_prime(n):
            num_verified += 1
            if best is None or n > best:
                best = n
    return num_claimed, num_verified, best


def _fmt_prime(n):
    s = str(n)
    if len(s) <= 40:
        return f"{s} ({len(s)} digits)"
    return f"{s[:16]}…{s[-12:]} ({len(s)} digits)"


def main():
    ap = argparse.ArgumentParser(description="autoprime immutable judge")
    ap.add_argument("--task", choices=["largest", "count"], default="largest")
    ap.add_argument("--time", type=int, default=TIME_BUDGET,
                    help="run budget in seconds (default 60)")
    ap.add_argument("--workspace", default="solve",
                    help="path to the agent-owned solve workspace")
    args = ap.parse_args()

    specs = detect_specs()
    print(f"[harness] {specs['chip']} | {specs['arch']} | "
          f"{specs['cores']} cores | gmp={specs['gmp_prefix'] or 'n/a'}",
          flush=True)
    print(f"[harness] task={args.task}  time_budget={args.time}s", flush=True)

    # Build (untimed). A build failure is treated like a crash, as in autoresearch.
    run_path = build(args.workspace, specs, args.task, args.time)

    # Run (timed).
    out_path, cutoff_size, solve_seconds = run_solver(
        run_path, args.workspace, args.task, args.time)

    # Verify + score (untimed but reported). Streamed so a huge run never lives
    # in memory at once; every claim is checked, taking as long as it needs to.
    t0 = time.monotonic()
    num_claimed, num_verified, best = score_run(out_path, cutoff_size)
    verify_seconds = time.monotonic() - t0
    try:
        os.remove(out_path)
    except OSError:
        pass

    if args.task == "largest":
        score = len(str(best)) if best is not None else 0
    else:  # count — distinct verified non-Mersenne primes, uncapped
        score = num_verified

    print("---")
    print(f"task:            {args.task}")
    print(f"score:           {score}")
    print(f"best_prime:      {_fmt_prime(best) if best is not None else 'none'}")
    print(f"num_verified:    {num_verified}")
    print(f"num_claimed:     {num_claimed}")
    print(f"solve_seconds:   {solve_seconds:.1f}")
    print(f"verify_seconds:  {verify_seconds:.1f}")
    print(f"cores:           {specs['cores']}")


if __name__ == "__main__":
    main()
