#!/bin/bash
# Advance one permutation null for a bounded window, with exactly one process.
#
# Background jobs in this environment are reclaimed after ~25 minutes, and a
# naive "launch and forget" leaves orphans that duplicate each other's work on
# the same checkpoint.  So: kill any existing runner first, start exactly one,
# and let the window's timeout end it.  Progress survives in the checkpoint.
set -u
SCRIPT="$1"; TARGET="$2"; SECS="${3:-540}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="${DH_ROOT:-$(dirname "$HERE")}/hebrew/results"
ps -eo pid,cmd | grep "[p]ython3 -u .*$SCRIPT" | awk '{print $1}' \
  | while read p; do kill -9 "$p" 2>/dev/null; done
sleep 1
timeout "$SECS" python3 -u "$HERE/$SCRIPT" "$TARGET" > "/tmp/$(basename $SCRIPT).log" 2>&1
CK=$([ "$SCRIPT" = "final_lobo.py" ] && echo final_lobo_null.csv || echo within_genre_null.csv)
echo "$(basename $SCRIPT): $(($(wc -l < "$RES/$CK")-1))/$TARGET draws"
tail -1 "/tmp/$(basename $SCRIPT).log"
