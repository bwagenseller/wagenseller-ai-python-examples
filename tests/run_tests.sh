#!/usr/bin/env bash
#
# Runs every test suite in this folder: each area's suite.sh (tests/<area>/.../suite.sh), in path order.
# Prints PASS / FAIL / SKIP per check and exits with the number of suites that had a failure.
#
# Usage:
#   tests/run_tests.sh                 every suite
#   tests/run_tests.sh --quick         skip the slow checks (the ~4-minute stream baselines)
#   tests/run_tests.sh --offline       skip everything that touches the internet
#   tests/run_tests.sh llm/llama/agent ai_tools     only the suites under these folders
#
# Machine-specific paths (models, conda env pythons, tool configs) come from tests/local_settings.json - copy
# tests/local_settings.example.json. A check whose setting is missing is skipped or fails with the key's name.
# AMADEO_TEST_SRC=/some/copy/of/src runs the suites against another copy of the library.
#
# Several suites load real model files (CPU is enough). While another process holds the GPU, run with
# CUDA_VISIBLE_DEVICES= so nothing tries to allocate GPU memory.

set -u
TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
export QUICK=0 OFFLINE=0
areas=()
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --offline) OFFLINE=1 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        -*) echo "unknown option: $arg (try --help)"; exit 64 ;;
        *) areas+=("$arg") ;;
    esac
done
[ ${#areas[@]} -eq 0 ] && areas=(".")

export LOG_DIR
LOG_DIR="$(mktemp -d -t amadeo-tests-XXXXXX)"
started=$(date +%s)
failed_suites=0
skipped_checks=0

cd "$TESTS_DIR" || exit 1
mapfile -t suites < <(for area in "${areas[@]}"; do find "$area" -name suite.sh; done | sed 's#^\./##' | sort -u)
if [ ${#suites[@]} -eq 0 ]; then
    echo "no suite.sh found under: ${areas[*]}"
    exit 64
fi

for suite in "${suites[@]}"; do
    echo "== $(dirname "$suite") =="
    # The suite's output is shown as it runs and kept, so its SKIP lines can be counted for the summary.
    bash "$suite" 2>&1 | tee "$LOG_DIR/suite_output.txt"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        failed_suites=$((failed_suites + 1))
    fi
    skipped_checks=$((skipped_checks + $(grep -c "^  SKIP " "$LOG_DIR/suite_output.txt")))
    echo
done

elapsed=$(( $(date +%s) - started ))
if [ "$failed_suites" -eq 0 ] && [ "$skipped_checks" -gt 0 ]; then
    # Not a plain pass: some checks did not run here (a missing setting, env or model), so they proved nothing.
    echo "NO FAILURES, BUT $skipped_checks CHECK(S) SKIPPED (${#suites[@]} suites, ${elapsed}s) - see the SKIP lines above"
    rm -rf "$LOG_DIR"
elif [ "$failed_suites" -eq 0 ]; then
    echo "ALL SUITES PASS (${#suites[@]} suites, ${elapsed}s)"
    rm -rf "$LOG_DIR"
else
    echo "$failed_suites of ${#suites[@]} suite(s) had failures (${elapsed}s) - logs kept in $LOG_DIR"
fi
exit "$failed_suites"
