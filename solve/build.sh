#!/usr/bin/env bash
# Default autoprime build. THIS FILE IS YOURS — rewrite it freely.
#
# The only contract: produce an executable ./run in this directory. That's it.
# Add frameworks, extra translation units, .metal shaders, .m (Objective-C),
# a Python entrypoint, model weights — whatever the current idea needs.
#
# The harness exports, for your use:
#   $AUTOPRIME_CORES        logical core count (e.g. 14)
#   $AUTOPRIME_GMP_PREFIX   Homebrew gmp prefix (headers/libs)
#   $AUTOPRIME_ARCH         machine arch (e.g. arm64)
#   $AUTOPRIME_TASK         largest | count
#   $AUTOPRIME_TIME         budget in seconds
#
# To unlock more of the machine, append linker flags here, e.g.:
#   -framework Accelerate                 # vDSP / BLAS / AMX matrix units, ML
#   -framework Metal -framework Foundation  # GPU compute (+ .metal/.m files)
set -euo pipefail

GMP_PREFIX="${AUTOPRIME_GMP_PREFIX:-/opt/homebrew/opt/gmp}"
GMP_FLAGS=(-lgmp)
if [ -n "$GMP_PREFIX" ] && [ -d "$GMP_PREFIX" ]; then
    GMP_FLAGS=(-I"$GMP_PREFIX/include" -L"$GMP_PREFIX/lib" -lgmp)
fi

# Apple clang wants -mcpu=native (NOT -march). Fall back if the host rejects it.
ARCH_FLAG="-mcpu=native"
if ! cc -x c -mcpu=native -c /dev/null -o /dev/null 2>/dev/null; then
    if cc -x c -mcpu=apple-m4 -c /dev/null -o /dev/null 2>/dev/null; then
        ARCH_FLAG="-mcpu=apple-m4"
    else
        ARCH_FLAG=""
    fi
fi

cc -O3 $ARCH_FLAG -std=c11 -o run solve.c -lm -lpthread "${GMP_FLAGS[@]}"
