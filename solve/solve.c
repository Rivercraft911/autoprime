#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <gmp.h>

#define MAX_THREADS 64
#define TARGET_DIGITS 3000
#define DIGIT_STRIDE 37

typedef struct {
    int id;
    int digits;
} worker_arg_t;

static pthread_mutex_t emit_lock = PTHREAD_MUTEX_INITIALIZER;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void emit_mpz(const mpz_t n) {
    pthread_mutex_lock(&emit_lock);
    fputs("PRIME ", stdout);
    mpz_out_str(stdout, 10, n);
    fputc('\n', stdout);
    fflush(stdout);
    pthread_mutex_unlock(&emit_lock);
}

static void *largest_worker(void *opaque) {
    const worker_arg_t *arg = (const worker_arg_t *)opaque;
    gmp_randstate_t rng;
    mpz_t seed, lo, span, offset, candidate, prime;

    gmp_randinit_mt(rng);
    mpz_inits(seed, lo, span, offset, candidate, prime, NULL);

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    mpz_set_ui(seed, (unsigned long)getpid());
    mpz_mul_2exp(seed, seed, 64);
    mpz_add_ui(seed, seed, (unsigned long)ts.tv_sec);
    mpz_mul_2exp(seed, seed, 32);
    mpz_add_ui(seed, seed, (unsigned long)ts.tv_nsec);
    mpz_mul_2exp(seed, seed, 16);
    mpz_add_ui(seed, seed, (unsigned long)arg->id);
    gmp_randseed(rng, seed);

    mpz_ui_pow_ui(lo, 10, (unsigned long)(arg->digits - 1));
    mpz_mul_ui(span, lo, 9);
    mpz_urandomm(offset, rng, span);
    mpz_add(candidate, lo, offset);
    mpz_setbit(candidate, 0);
    mpz_nextprime(prime, candidate);

    if (mpz_sizeinbase(prime, 10) == (size_t)arg->digits) emit_mpz(prime);

    mpz_clears(seed, lo, span, offset, candidate, prime, NULL);
    gmp_randclear(rng);
    return NULL;
}

static void run_largest(double budget) {
    int cores = 14;
    const char *env_cores = getenv("AUTOPRIME_CORES");
    if (env_cores && atoi(env_cores) > 0) cores = atoi(env_cores);
    if (cores > MAX_THREADS) cores = MAX_THREADS;

    pthread_t threads[MAX_THREADS];
    worker_arg_t args[MAX_THREADS];
    for (int i = 0; i < cores; i++) {
        args[i].id = i;
        args[i].digits = TARGET_DIGITS + i * DIGIT_STRIDE;
        if (pthread_create(&threads[i], NULL, largest_worker, &args[i]) == 0) {
            pthread_detach(threads[i]);
        }
    }

    double deadline = now_seconds() + budget - 0.2;
    while (now_seconds() < deadline) {
        struct timespec nap = {.tv_sec = 0, .tv_nsec = 20000000};
        nanosleep(&nap, NULL);
    }
}

static void run_count(void) {
    puts("PRIME 2");
    puts("PRIME 3");
    puts("PRIME 5");
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *task = "largest";
    double budget = 60.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--task") && i + 1 < argc) task = argv[++i];
        else if (!strcmp(argv[i], "--time") && i + 1 < argc) budget = atof(argv[++i]);
    }

    if (!strcmp(task, "count")) run_count();
    else run_largest(budget);
    return 0;
}
