#!/usr/bin/env bash
# Block until the model server answers, then report what it is actually serving.
#
# The second part matters as much as the first: if MODEL and the model the server
# loaded disagree, every result gets attributed to the wrong model — which is
# worse than an outright failure, because it looks like data.

set -uo pipefail

PORT="${1:-8000}"
EXPECT="${2:-}"
TRIES="${TRIES:-240}"
SLEEP="${SLEEP:-10}"

echo "Waiting for the endpoint on port ${PORT} — a first run pulls weights, allow several minutes."

for _ in $(seq 1 "$TRIES"); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo
    served=$(curl -s "http://localhost:${PORT}/v1/models" \
      | python3 -c 'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null)
    echo "Endpoint is up. Serving: ${served:-<could not parse>}"

    if [ -n "$EXPECT" ] && [ -n "$served" ] && [ "$served" != "$EXPECT" ]; then
      echo
      echo "FAIL  the server is serving '${served}' but MODEL is '${EXPECT}'."
      echo "      Results would be attributed to the wrong model. Fix before running anything."
      exit 1
    fi
    exit 0
  fi
  printf '.'
  sleep "$SLEEP"
done

echo
echo "FAIL  endpoint did not come up in $((TRIES * SLEEP / 60)) minutes."
echo "      Check the logs:  make serve-logs"
exit 1
