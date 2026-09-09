#!/usr/bin/env bash
# Exercise the REAL vllm backend against a fake endpoint, across every response
# scenario. No GPU: this is the code between us and the model.
#
# Each scenario declares what SHOULD happen. An earlier version only checked the
# exit code, that the output parsed, and that no event fell outside the video —
# which a hallucinated timestamp CLAMPED to the window edge satisfies perfectly.
# It therefore passed both before and after the bug it was supposed to catch.
# Assertions that cannot fail are decoration.
set -uo pipefail
cd "$(dirname "$0")/.."

VIDEO="${VIDEO_IN:-/data/synthetic/box-crossing.mp4}"
Q="${QUERIES:-a red box enters from the left}"
PORT="${VLLM_PORT:-8000}"

#          scenario:expectation   min:max events, or 'any'
EXPECT="think:1:9 plain:1:9 fenced:1:9 prose:1:9 empty:0:0 truncated:0:0 \
refusal:0:0 hallucinate:0:0 malformed:0:0 mixed:0:9"
[ $# -gt 0 ] && EXPECT="$*"

fails=0
printf "%-12s %-7s %-9s %-7s %s\n" SCENARIO EVENTS EXPECTED BOUNDS RESULT
for spec in $EXPECT; do
  sc="${spec%%:*}"; rest="${spec#*:}"; lo="${rest%%:*}"; hi="${rest##*:}"

  # Remove first, then start. --force-recreate alone is not enough: a leftover
  # container can still hold the port, so the new one fails to bind and the OLD
  # scenario keeps answering. Every case would then be tested against whichever
  # scenario happened to start first, while the harness reported real numbers.
  docker compose rm -sf fake-vllm >/dev/null 2>&1
  SCENARIO="$sc" docker compose up -d fake-vllm >/dev/null 2>&1
  for _ in $(seq 1 30); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  # Confirm the endpoint is serving the scenario we asked for, rather than
  # assuming the restart took effect.
  got=$(docker compose logs fake-vllm 2>/dev/null | grep -o 'scenario [a-z0-9]*' | tail -1 | awk '{print $2}')
  if [ -n "$got" ] && [ "$got" != "$sc" ]; then
    printf "%-12s %-7s %-9s %-7s %s\n" "$sc" "-" "$lo-$hi" "-" "FAIL endpoint serving '$got'"
    fails=$((fails+1)); continue
  fi

  err=$(mktemp)
  out=$(docker compose run --rm --no-deps -e BASE_URL=http://fake-vllm:8000/v1 \
        finder video-reasoning run "$VIDEO" --queries "$Q" \
        --backend vllm --base-url http://fake-vllm:8000/v1 --quiet 2>"$err")
  rc=$?

  if [ $rc -ne 0 ]; then
    printf "%-12s %-7s %-9s %-7s %s\n" "$sc" "-" "$lo-$hi" "-" "FAIL rc=$rc"
    tail -2 "$err" | sed 's/^/               /'; fails=$((fails+1)); rm -f "$err"; continue
  fi
  rm -f "$err"

  read -r n bad <<<"$(echo "$out" | python3 -c '
import json,sys
d=json.load(sys.stdin); dur=d["duration_s"]
bad=sum(1 for e in d["events"] if e["start_s"]<0 or e["end_s"]>dur+1e-6)
print(len(d["events"]), bad)' 2>/dev/null || echo "? ?")"

  res="ok"
  if [ "$n" = "?" ]; then res="FAIL unparseable"
  elif [ "$bad" != "0" ]; then res="FAIL $bad out of bounds"
  elif [ "$n" -lt "$lo" ] || [ "$n" -gt "$hi" ]; then res="FAIL expected $lo-$hi"
  fi
  [ "$res" = "ok" ] || fails=$((fails+1))
  printf "%-12s %-7s %-9s %-7s %s\n" "$sc" "$n" "$lo-$hi" "$bad" "$res"
done

docker compose stop fake-vllm >/dev/null 2>&1
docker compose rm -f fake-vllm >/dev/null 2>&1
echo
[ $fails -eq 0 ] && echo "all scenarios behaved as specified" \
                 || echo "$fails scenario(s) misbehaved"
exit $fails
