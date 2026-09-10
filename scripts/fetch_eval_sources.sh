#!/usr/bin/env bash
# Fetch the SOURCE footage that data/eval/labels.json was cut from.
#
# The labelled clips are not in the repository -- 400 MB of video would bloat
# every clone forever, and it is not ours to keep republishing when the source is
# public. What IS in the repository is the labels plus, per clip, the source file
# name, the trim offset and the duration. That is enough to rebuild the eval set
# byte-for-byte, which is the property that matters: a reviewer can reproduce every
# number we report, from a published dataset, without taking our copy on trust.
#
#   bash scripts/fetch_eval_sources.sh data/eval/labels.json data/meva-annotated
set -euo pipefail

LABELS="${1:-data/eval/labels.json}"
OUT="${2:-data/meva-annotated}"
BUCKET="${MEVA_S3:-s3://mevadata-public-01}"
DROP="${MEVA_DROP:-drops-123-r13}"
AWS=(aws s3 --no-sign-request --no-cli-pager --only-show-errors)

[ -f "$LABELS" ] || { echo "FAIL  no labels at ${LABELS}" >&2; exit 1; }
mkdir -p "$OUT"

sources=$(python3 -c "
import json,sys
print('\n'.join(sorted({e['source'] for e in json.load(open(sys.argv[1]))})))" "$LABELS")
want=$(printf '%s\n' "$sources" | grep -c . || true)
echo "labels.json needs ${want} source clip(s)"
echo

got=0; have=0; missing=()
while read -r fname; do
  [ -n "$fname" ] || continue
  if [ -f "${OUT}/${fname}" ]; then
    printf "  have  %s\n" "$fname"; have=$((have+1)); continue
  fi
  # stem without the release suffix and extension, e.g. ...admin.G329
  stem="${fname%.*}"; stem="${stem%.r13}"
  day="${stem%%.*}"
  h_start=$(echo "$stem" | cut -d. -f2 | cut -d- -f1)
  h_end=$(echo "$stem" | cut -d. -f3 | cut -d- -f1)
  found=""
  # End hour first: MEVA files an hour-crossing clip under the hour it ENDS in.
  for hour in "$h_end" "$h_start"; do
    if "${AWS[@]}" cp "${BUCKET}/${DROP}/${day}/${hour}/${fname}" "${OUT}/" 2>/dev/null; then
      found=1; break
    fi
  done
  if [ -n "$found" ]; then
    printf "  OK    %-52s %sMB\n" "$fname" "$(du -m "${OUT}/${fname}" | cut -f1)"
    got=$((got+1))
  else
    printf "  MISS  %s\n" "$fname"; missing+=("$fname")
  fi
done <<< "$sources"

echo
echo "${have} already present, ${got} fetched, ${#missing[@]} missing"
if [ "${#missing[@]}" -gt 0 ]; then
  echo "FAIL  could not fetch:" >&2
  printf '        %s\n' "${missing[@]}" >&2
  exit 1
fi
