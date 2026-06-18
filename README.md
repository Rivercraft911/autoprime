# autoprime

**Work in progress.**

A riff on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Autoresearch turns an AI agent loose overnight to improve a tiny training run; autoprime points the same loop at **prime numbers**, and dreams of climbing, run after run, towards greatness.

The agent owns a solver workspace; a fixed judge verifies every claim. The fun is range: invent an algorithm, optimize a classic sieve, use every core, reach for libgmp or the GPU, or just guess enormous candidates and check them. The judge doesn't care *how*, only that the primes are real, non-Mersenne, and found in time.

## Run

```bash
pip install sympy
python3 harness.py --task largest   # biggest non-Mersenne prime in 60s
python3 harness.py --task count     # most distinct primes in 60s (uncapped, rigorously verified)
```

`harness.py` builds `solve/`, runs it for the budget, verifies every `PRIME <n>` it printed, and reports a `score`. Then point an agent at `program.md` and let it loop.
