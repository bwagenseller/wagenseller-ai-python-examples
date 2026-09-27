# Shared functions for the tests' suite.sh files. Sourced, never run directly.
#
# A suite.sh sources this, then calls:
#   run LABEL COMMAND...                       one check; PASS/FAIL from its exit code, output kept in a log
#   compare LABEL BASELINE COMMAND...          runs a capture that writes to "$OUT", then diffs it against BASELINE
#   skip LABEL REASON                          records a check that could not run here (not a failure)
#   setting KEY                                prints a local setting (tests/local_settings.json), or nothing
#
# Environment it reads (set by run_tests.sh, or by hand when a suite.sh is run on its own):
#   QUICK=1      skip the slow checks (each suite decides which those are)
#   OFFLINE=1    skip every check that touches the internet
#   LOG_DIR      where logs go (a fresh temporary folder when unset)
#
# At the end a suite calls 'finish', which prints its totals and exits with its failure count.

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
SRC="${AMADEO_TEST_SRC:-$REPO_ROOT/src}"
QUICK="${QUICK:-0}"
OFFLINE="${OFFLINE:-0}"
LOG_DIR="${LOG_DIR:-$(mktemp -d -t amadeo-tests-XXXXXX)}"
mkdir -p "$LOG_DIR"
failures=0
passes=0
skips=0

setting() {
    # A local setting with '~' expanded, or an empty string when it is not set.
    python3 "$TESTS_DIR/testlib/settings.py" --get "$1" 2>/dev/null
}

run() {
    # $1 = label, the rest = the command. The command's output goes to a log; the last lines are shown on failure.
    local label="$1"; shift
    local log="$LOG_DIR/$(echo "$label" | tr -c 'A-Za-z0-9' '_' | cut -c1-80).log"
    local t0
    t0=$(date +%s)
    if "$@" >"$log" 2>&1; then
        echo "  PASS  $label ($(( $(date +%s) - t0 ))s)"
        passes=$((passes + 1))
    else
        echo "  FAIL  $label ($(( $(date +%s) - t0 ))s) - last lines of $log:"
        grep -v " - INFO - \| - DEBUG - " "$log" | tail -15 | sed 's/^/        /'
        failures=$((failures + 1))
    fi
}

compare() {
    # $1 = label, $2 = baseline file, the rest = a command that writes its capture to "$OUT".
    local label="$1" baseline="$2"; shift 2
    if [ ! -f "$baseline" ]; then
        echo "  FAIL  $label - baseline $baseline is missing"
        failures=$((failures + 1))
        return
    fi
    run "$label (capture)" "$@"
    if [ -f "$OUT" ] && diff -q "$baseline" "$OUT" >/dev/null 2>&1; then
        echo "  PASS  $label matches $(basename "$baseline")"
        passes=$((passes + 1))
    else
        echo "  FAIL  $label differs from $(basename "$baseline"):"
        diff "$baseline" "$OUT" 2>&1 | head -30 | sed 's/^/        /'
        failures=$((failures + 1))
    fi
}

skip() {
    # $1 = label, $2 = why it did not run.
    echo "  SKIP  $1 ($2)"
    skips=$((skips + 1))
}

need_python() {
    # $1 = the settings key of a conda env's python. Prints it if it exists; otherwise prints nothing.
    local py
    py="$(setting "$1")"
    if [ -n "$py" ] && [ -x "$py" ]; then
        echo "$py"
    fi
}

finish() {
    echo "  -- $passes passed, $failures failed, $skips skipped"
    exit "$failures"
}
