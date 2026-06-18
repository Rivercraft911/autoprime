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
  4. VERIFY every claimed prime independently with sympy, rejecting Mersenne
     numbers and anything that isn't actually prime. This is untimed and
     rigorous (deterministic for all n < 2**64) — it takes as long as it needs
     to be certain, and is streamed so even a multi-GB run stays in bounds.
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

import sympy

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

    sympy.isprime is deterministic for every n < 2**64 (BPSW), which covers any
    prime a `count` solver can realistically enumerate — so a verified claim is
    certain, not merely probable. This check is deliberately untimed: the judge
    takes as long as it needs to be sure and never trades rigor for speed.
    """
    return n > 1 and not is_mersenne(n) and sympy.isprime(n)


def score_run(out_path, cutoff_size):
    """Stream the capture file up to the deadline cutoff and score it.

    One pass: dedup distinct `PRIME <n>` claims, verify each rigorously, reject
    Mersenne, and track how many verified and the maximum. Reading line by line
    (rather than slurping the whole file) keeps the judge's footprint bounded by
    the set of distinct claims, so `count` has no artificial ceiling — only
    CLAIM_SAFETY_LIMIT, a far-off memory guard, caps what we will remember.

    Only bytes present at the official deadline count: we stop at cutoff_size and
    ignore any final partial (non-newline-terminated) line.
    """
    seen = set()
    num_verified = 0
    best = None
    remaining = cutoff_size
    with open(out_path, "rb") as f:
        for line in f:
            if len(line) > remaining:
                break  # line straddles the deadline cutoff — drop the partial tail
            remaining -= len(line)
            if not line.endswith(b"\n"):
                break  # unterminated final line — ignore it
            m = PRIME_RE.match(line)
            if not m:
                continue  # noise and truncated lines are ignored
            n = int(m.group(1))
            if n in seen:
                continue
            seen.add(n)
            if is_genuine_prime(n):
                num_verified += 1
                if best is None or n > best:
                    best = n
            if len(seen) >= CLAIM_SAFETY_LIMIT:
                break
    return len(seen), num_verified, best


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
