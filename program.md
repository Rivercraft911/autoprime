# autoprime

You have the whole machine to play with: every core, the GPU, the matrix units,
any library or model you can build. There is no single right approach here;
follow your curiosity. The standing goal is to keep improving, forever.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (for example
   `jun18`). The branch `autoprime/<tag>` must not already exist; this is a
   fresh run.
2. **Create the branch**: `git checkout -b autoprime/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full
   context:
   - `README.md` - repository context and task definitions.
   - `harness.py` - fixed build/run/verify/score judge. Do not modify.
   - `solve/` - the workspace you modify. Everything in here is fair game.
4. **Verify dependencies**: `python3 -c "import sympy"` must work. If it does
   not, ask the human to install `requirements.txt`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row:

   ```text
   commit	task	score	status	description
   ```

6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs for a fixed wall-clock budget, normally 60 seconds. You
launch it as:

```bash
python3 harness.py --task largest > run.log 2>&1
```

or:

```bash
python3 harness.py --task count > run.log 2>&1
```

**What you CAN do:**

- Modify anything under `solve/`.
- Rewrite `solve/build.sh`.
- Add C, C++, Objective-C, Metal shaders, Python helpers, model weights,
  generated lookup tables, or any other committed workspace files.
- Use libgmp, pthreads, GCD/libdispatch, NEON, Metal, Accelerate/AMX, and any
  system library available on the machine.

**What you CANNOT do:**

- Modify `harness.py`. It is read-only and is the ground-truth verifier.
- Count unverified output. Only `PRIME <decimal>` lines accepted by the harness
  score.
- Rely on Mersenne primes. The harness rejects numbers of the form `2^k - 1`.

The goal is simple: maximize `score` for the chosen task.

- `largest`: score is the decimal digit count of the largest verified
  non-Mersenne prime.
- `count`: score is the number of distinct verified non-Mersenne primes. It is
  **uncapped** — an exhaustive-enumeration climb with no ceiling, so the winning
  move is to deterministically find *every* prime up to the highest bound you
  can reach (no probabilistic luck). Climb it forever.

The time budget is fixed, so runs are comparable. Build time and verification
time are reported but do not count against the solver's budget — verification is
rigorous and takes as long as it needs to be certain every counted prime is real.

## Creativity Charter

Autoresearch prizes simple training-code hill climbing. Autoprime prizes range.
Across iterations, swing between lanes:

1. **Invent**: try a novel search, residue-class wheel, k-tuple pattern,
   prime-gap heuristic, or custom sieve shape.
2. **Optimize**: implement a known method better: segmented sieve,
   Miller-Rabin, BPSW, ECPP, cache-blocking, SIMD, threads, GPU kernels.
3. **Get weird**: probabilistically guess huge candidates and verify them,
   train a tiny statistical model to propose prime-rich regions, or brute-force
   strange ranges on the GPU.

Variety is the feature. Still, the branch only advances when the score improves.

## Output Format

The harness prints a grep-able block:

```text
---
task:            largest
score:           1234
best_prime:      4928...1007 (1234 digits)
num_verified:    17
num_claimed:     21
solve_seconds:   60.0
verify_seconds:  3.2
cores:           14
```

Extract the metric with:

```bash
grep "^score:" run.log
```

If the grep output is empty, the run crashed. Read the tail:

```bash
tail -n 50 run.log
```

## Logging Results

When an experiment is done, log it to `results.tsv` as tab-separated values.
Do not commit `results.tsv`.

```text
commit	task	score	status	description
a1b2c3d	largest	13	keep	baseline trial division
b2c3d4e	largest	512	keep	gmp random odd candidates plus Miller-Rabin
c3d4e5f	count	0	crash	segmented sieve race in output buffer
```

Status must be one of `keep`, `discard`, or `crash`.

## The Experiment Loop

The experiment runs on a dedicated branch such as `autoprime/jun18`.

LOOP FOREVER:

1. Look at the git state: current branch and commit.
2. Pick one experimental idea.
3. Modify `solve/`.
4. Commit the change.
5. Run the experiment:

   ```bash
   python3 harness.py --task <task> > run.log 2>&1
   ```

6. Read the score:

   ```bash
   grep "^score:" run.log
   ```

7. If the grep output is empty, inspect `tail -n 50 run.log`. Fix obvious
   typos or build bugs and rerun; if the idea is fundamentally broken, log it
   as `crash`.
8. Record the result in `results.tsv`.
9. If the score improved over the branch best for that task, keep the commit.
10. If the score is equal or worse, `git reset` back to where you started.

**Timeout**: Each experiment should take about 60 seconds plus build and verify
overhead. If a run greatly exceeds that, kill it and treat it as a failure.

**Crashes**: If a run crashes because of a small mistake, fix it and rerun. If
the idea itself is broken, log `crash`, reset, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the
human if you should continue. Do NOT ask "should I keep going?" or "is this a
good stopping point?". The human might be asleep or away from the computer and
expects you to continue working indefinitely until manually stopped. You are
autonomous. If you run out of ideas, think harder, reread the in-scope files,
try combining previous near-misses, or try a more radical direction. The loop
runs until the human interrupts you, period.

