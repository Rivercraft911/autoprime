/*
 * autoprime baseline solver.
 *
 * This is YOUR file (and the rest of solve/). The harness only requires that
 * build.sh produces an executable ./run, and that ./run prints lines:
 *
 *     PRIME <decimal>\n
 *
 * one per prime found, flushed as you go. The harness independently verifies
 * every claim (genuine prime, non-Mersenne) and ignores everything else.
 *
 * Two tasks, selected with --task:
 *   largest : maximize the number of digits of the biggest prime you find.
 *   count   : maximize how many distinct primes you find.
 * --time <seconds> is your wall-clock budget (the harness hard-kills you a few
 * seconds after it elapses, so stop yourself a hair early).
 *
 * This baseline is deliberately simple and obviously correct: single-threaded
 * trial division. There is enormous headroom above it. The escalation menu:
 *   - segmented Sieve of Eratosthenes, wheel factorization
 *   - Miller-Rabin / BPSW / ECPP for fast primality
 *   - pthreads or GCD across all cores  (see $AUTOPRIME_CORES)
 *   - NEON SIMD (on by default for arm64), cache-blocking
 *   - libgmp (already linked) for thousand-digit `largest` candidates
 *   - the Metal GPU and the Accelerate/AMX matrix units (edit build.sh)
 *   - get weird: probabilistically guess giant candidates; train a tiny model
 *     to predict prime-rich regions and verify its guesses.
 * How you find them is entirely up to you. Only verified primes score.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Trial division. Obviously correct, intentionally unoptimized. */
static int is_prime_u64(uint64_t n) {
    if (n < 2) return 0;
    if (n < 4) return 1;            /* 2, 3 */
    if ((n & 1) == 0) return 0;
    if (n % 3 == 0) return 0;
    for (uint64_t i = 5; i <= n / i; i += 6) {
        if (n % i == 0) return 0;
        if (n % (i + 2) == 0) return 0;
    }
    return 1;
}

static void emit(uint64_t n) {
    printf("PRIME %llu\n", (unsigned long long)n);
    fflush(stdout);
}

/* count: stream every prime upward from 2 until the budget runs out. */
static void run_count(double deadline) {
    uint64_t checked = 0;
    for (uint64_t n = 2;; n++) {
        if (is_prime_u64(n)) emit(n);
        if ((++checked & 0x3FFF) == 0 && now_seconds() >= deadline) return;
    }
}

/* largest: walk upward from a modest start; the harness keeps the max. */
static void run_largest(double deadline) {
    uint64_t n = 1000000000000ULL; /* ~1e12; trial division is still feasible */
    n |= 1ULL;                     /* make odd; we then step by 2 */
    uint64_t checked = 0;
    for (;; n += 2) {
        if (is_prime_u64(n)) emit(n);
        if ((++checked & 0x3FF) == 0 && now_seconds() >= deadline) return;
    }
}

int main(int argc, char **argv) {
    const char *task = "largest";
    double budget = 60.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--task") && i + 1 < argc) task = argv[++i];
        else if (!strcmp(argv[i], "--time") && i + 1 < argc) budget = atof(argv[++i]);
    }

    /* Stop a touch early so a full final line is flushed before the hard-kill. */
    double deadline = now_seconds() + budget - 0.5;

    if (!strcmp(task, "count")) run_count(deadline);
    else run_largest(deadline);
    return 0;
}
