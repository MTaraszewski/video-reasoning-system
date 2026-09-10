#!/usr/bin/env bash
# Fetch a hand-chosen shortlist of MEVA clips, selected by `screen_geom.py` for
# actor size and by hand for event coverage.
#
# This exists rather than a filter on fetch_meva.sh because the selection is a
# JUDGEMENT — which events the eval set should cover — and judgements belong in a
# reviewable file, not in a filter string.
#
# Two path facts, both learned by having URLs 404:
#
#   1. Video filenames carry a release suffix the annotation names do not:
#        annotation  ...bus.G340.activities.yml
#        video       ...bus.G340.r13.avi
#   2. A clip that CROSSES AN HOUR BOUNDARY is filed under its END hour:
#        .../2018-03-15/14/2018-03-15.14-50-00.14-55-00.school.G421.r13.avi
#        .../2018-03-15/15/2018-03-15.14-55-00.15-00-00.school.G421.r13.avi
#      Deriving the hour from the start time loses exactly the boundary-crossing
#      clips — 2 of 6 in the first shortlist — while the rest succeed, so the run
#      looks fine unless you count what came back.
#
#   bash scripts/fetch_meva_shortlist.sh data/meva-index/shortlist.json data/meva-annotated
set -euo pipefail

LIST="${1:-data/meva-index/shortlist.json}"
OUT="${2:-data/meva-annotated}"
BUCKET="${MEVA_S3:-s3://mevadata-public-01}"
DROP="${MEVA_DROP:-drops-123-r13}"
RELEASE="${MEVA_RELEASE:-r13}"
AWS=(aws s3 --no-sign-request --no-cli-pager --only-show-errors)

[ -f "$LIST" ] || { echo "FAIL  no shortlist at ${LIST}" >&2; exit 1; }
mkdir -p "${OUT}/annotations"

stems=$(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))))" "$LIST")
want=$(printf '%s\n' "$stems" | grep -c . || true)
echo "Shortlist: ${want} clip(s) from ${LIST}"
echo

got=0; missing=()
while read -r stem; do
  [ -n "$stem" ] || continue
  day="${stem%%.*}"
  h_start=$(echo "$stem" | cut -d. -f2 | cut -d- -f1)
  h_end=$(echo "$stem" | cut -d. -f3 | cut -d- -f1)

  found=""
  # End hour first: it is the correct one whenever the two differ.
  for hour in "$h_end" "$h_start"; do
    for name in "${stem}.${RELEASE}" "${stem}"; do
      for ext in avi mp4; do
        src="${BUCKET}/${DROP}/${day}/${hour}/${name}.${ext}"
        if "${AWS[@]}" cp "$src" "${OUT}/" 2>/dev/null; then
          found="${name}.${ext}"
          # The annotation is indexed by START hour under examples/, a different
          # convention from the video path. Try both there too.
          for ah in "$h_start" "$h_end"; do
            "${AWS[@]}" cp \
              "${BUCKET}/examples/annotations/${day}/${ah}/${stem}.activities.yml" \
              "${OUT}/annotations/" 2>/dev/null && break
          done
          break 3
        fi
      done
    done
  done

  if [ -n "$found" ]; then
    sz=$(du -m "${OUT}/${found}" | cut -f1)
    printf "  OK    %-50s %sMB\n" "$found" "$sz"
    got=$((got + 1))
  else
    printf "  MISS  %-50s tried hours %s and %s\n" "$stem" "$h_end" "$h_start"
    missing+=("$stem")
  fi
done <<< "$stems"

cat > "${OUT}/ATTRIBUTION.txt" <<'ATTR'
MEVA (Multiview Extended Video with Activities)
Kitware Inc., collected for the IARPA Deep Intermodal Video Analytics programme.
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
https://mevadata.org/

Attribution is required for these clips and for any derivative of them,
including trimmed excerpts.
ATTR

echo
echo "${got}/${want} clip(s) fetched into ${OUT}"
# Count what came back, always. A partial fetch that exits 0 is how the earlier
# script reported success for an empty directory.
if [ "${#missing[@]}" -gt 0 ]; then
  echo "FAIL  ${#missing[@]} clip(s) not found:" >&2
  printf '        %s\n' "${missing[@]}" >&2
  exit 1
fi
ATTR_OK=1
