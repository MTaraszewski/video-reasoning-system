#!/usr/bin/env bash
# Where is a running `make eval STRATEGY=states` right now?
#
# The eval calls find_events with progress off, so it prints one line per CLIP and
# nothing in between -- 30+ silent minutes. This reconstructs the position from
# the model server's own request log instead, without touching the running job.
#
# It works because the polling order is deterministic (core.py `_poll_states`):
# for each timestep, one call per subject. So calls/subjects is the poll index,
# and poll index x step_s is the second of video being looked at.
#
# Counting starts from the eval container's start time, so earlier runs on the
# same server are not included.
#
#   scripts/eval_progress.sh [interval_seconds]
#
# With no argument it prints once. With one, it repeats until interrupted.
set -uo pipefail

SUBJECTS=${SUBJECTS:-4}       # distinct state pairs being polled
POLLS=${POLLS:-119}           # timesteps per clip: (duration - span_s) / step_s
CLIPS=${CLIPS:-4}             # EVAL_LIMIT
STEP_S=${STEP_S:-1.0}
INTERVAL=${1:-0}

per_clip=$(( SUBJECTS * POLLS ))
total=$(( per_clip * CLIPS ))

container=$(docker ps --filter "name=finder-run" --format '{{.Names}}' | head -1)
if [ -z "$container" ]; then
  echo "no eval container running (looked for a name containing 'finder-run')"
  exit 1
fi
started=$(docker inspect -f '{{.State.StartedAt}}' "$container")

show() {
  local n clip within poll sec pct
  n=$(docker compose logs --since "$started" vllm 2>/dev/null \
      | grep -c "chat/completions")
  [ "$n" -eq 0 ] && { echo "no calls yet"; return; }
  clip=$(( n / per_clip ))
  within=$(( n % per_clip ))
  poll=$(( within / SUBJECTS ))
  sec=$(echo "$poll $STEP_S" | awk '{printf "%.1f", $1 * $2}')
  pct=$(echo "$n $total" | awk '{printf "%.1f", 100 * $1 / $2}')
  [ "$clip" -ge "$CLIPS" ] && { echo "$n/$total calls — finishing"; return; }
  printf 'clip %d/%d   t=%ss of video   %d/%d calls   %s%%\n' \
    "$((clip + 1))" "$CLIPS" "$sec" "$n" "$total" "$pct"
}

if [ "$INTERVAL" -gt 0 ] 2>/dev/null; then
  while show; do sleep "$INTERVAL"; done
else
  show
fi
