#!/bin/bash
#
# Move finished SLURM .out logs into the per-run output_dir embedded in each log.
#
# Each run prints "Task .../...: ... output_dir=<dir> ..." (see run_tas.sh /
# run_entropy_experiment.sh). This script parses that <dir> out of every matching
# .out file and moves the log there, but SKIPS any log whose job is still active
# in the queue, so it's safe to run while an array is mid-flight.
#
# Usage:
#   scripts/collect_out_logs.sh [glob]      # default glob: tas_search-*.out
#   scripts/collect_out_logs.sh -n [glob]   # dry run: show actions, move nothing
#
# Examples:
#   scripts/collect_out_logs.sh                       # collect tas_search logs
#   scripts/collect_out_logs.sh 'entropy_exp-*.out'   # collect entropy logs
#   scripts/collect_out_logs.sh -n                    # preview only

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where it's invoked from

DRY=0
if [ "${1:-}" = "-n" ] || [ "${1:-}" = "--dry-run" ]; then
    DRY=1
    shift
fi
GLOB="${1:-tas_search-*.out}"

# Individual job ids still in the queue (any state). Logs for these are still
# being written, so we leave them alone.
ACTIVE=$( { squeue -u "$USER" --array -o "%A" 2>/dev/null || true; } \
    | grep -E '^[0-9]+$' | sort -u | tr '\n' '|' | sed 's/|$//' || true)
[ -n "$ACTIVE" ] && echo "Active job ids (skipped): ${ACTIVE//|/ }"
[ "$DRY" -eq 1 ] && echo "(dry run — nothing will be moved)"
echo

shopt -s nullglob
moved=0; skipped=0
for f in $GLOB; do
    id=$(echo "$f" | grep -oP '\d+' | head -1)
    if [ -n "$ACTIVE" ] && [ -n "$id" ] && echo "$id" | grep -qE "^($ACTIVE)$"; then
        echo "SKIP (active): $f"; skipped=$((skipped+1)); continue
    fi
    dest=$(grep -ohP 'output_dir=\K[^ ]+' "$f" | head -1)
    if [ -z "$dest" ]; then
        echo "SKIP (no output_dir in log): $f"; skipped=$((skipped+1)); continue
    fi
    if [ ! -d "$dest" ]; then
        echo "SKIP (dest missing: $dest): $f"; skipped=$((skipped+1)); continue
    fi
    if [ "$DRY" -eq 1 ]; then
        echo "WOULD MOVE: $f -> $dest/"
    else
        mv "$f" "$dest/"
        echo "MOVED: $f -> $dest/"
    fi
    moved=$((moved+1))
done

echo
tag=""; [ "$DRY" -eq 1 ] && tag="(dry run) "
echo "Done. ${tag}moved=$moved skipped=$skipped"
