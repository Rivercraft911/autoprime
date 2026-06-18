# autoprime

Autoprime is a prime-number hill-climbing playground inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch). Where
autoresearch turns an AI agent loose overnight to improve a tiny training run,
autoprime gives the agent a mutable solver workspace, a fixed verifier, a fixed
time budget, and one scalar score: find better primes, forever.

The fun is range. A run can invent a search strategy, optimize a classic sieve
or primality test, use all CPU cores, reach for libgmp, Metal, or Accelerate, or
get strange with probabilistic candidate generation. The judge does not care how
the number was found. It only scores verified, non-Mersenne primes emitted within
the solver budget.

## How It Works

- **`harness.py`** - immutable build/run/verify/score judge. It builds the
  solver, runs it for the requested budget, independently verifies claimed
  primes with SymPy, rejects Mersenne primes, and prints the score block.
- **`solve/`** - mutable agent-owned workspace. The default baseline is plain C
  trial division plus a `build.sh`, but agents can rewrite it completely.
- **`program.md`** - autonomous experiment loop instructions adapted from
  autoresearch.

The solver protocol is intentionally tiny:

```text
PRIME <decimal>
```

Each complete line is one claimed candidate. The harness deduplicates and
verifies every claim.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 harness.py --task largest
python3 harness.py --task count
```

Both tasks default to a 60 second solver budget. For a smoke test:

```bash
python3 harness.py --task count --time 1
```

## Tasks

- **`largest`**: maximize the decimal digit count of the largest verified
  non-Mersenne prime found in the time budget.
- **`count`**: maximize the number of distinct verified non-Mersenne primes
  found in the time budget.

The summary is grep-able:

```text
---
task:            largest
score:           13
best_prime:      1000000000039 (13 digits)
num_verified:    42
num_claimed:     45
solve_seconds:   60.0
verify_seconds:  0.1
cores:           14
```

## Machine Surface

The harness exports useful facts to `solve/build.sh`:

- `AUTOPRIME_CORES`
- `AUTOPRIME_GMP_PREFIX`
- `AUTOPRIME_ARCH`
- `AUTOPRIME_TASK`
- `AUTOPRIME_TIME`

The default build uses Apple clang, `-mcpu=native` when available, pthreads, and
libgmp when Homebrew exposes it. Agents are encouraged to rewrite the build to
use frameworks such as Metal, Foundation, Dispatch, or Accelerate.

## Ground Rule

Only `harness.py` decides what counts. The solver may use any strategy, but
bogus composites, duplicated claims, Mersenne primes such as `7`, and output
after the official budget do not improve the score.
