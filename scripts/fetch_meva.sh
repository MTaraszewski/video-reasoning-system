#!/usr/bin/env bash
# Fetch MEVA footage that actually contains events.
#
# MEVA is CC BY 4.0 — Kitware, collected for the IARPA DIVA programme — and is
# hosted free via the AWS Public Dataset Program. No credentials, no agreement,
# hence --no-sign-request. ATTRIBUTION IS REQUIRED wherever these clips or any
# derivative of them are used.
#
# WHY SELECTION MATTERS MORE THAN IT LOOKS
#
# MEVA released 328 hours of continuous surveillance, of which only ~22-27 hours
# carry public annotations. It exists for the NIST ActEV challenge, where the task
# is finding rare activity in long, mostly-empty footage — so most of the corpus
# is DELIBERATELY uneventful. That is the domain, not a flaw in the dataset.
#
# An earlier version of this script selected by filename duration alone, taking
# the first N alphabetically. It returned twelve camera views of a static car
# park: two minutes each of nothing, with no event to label. Correct code,
# useless data. Selection is therefore by ACTIVITY:
#
#   examples   121 pre-cut clips named for the activity they contain, e.g.
#              ex012-vehicle-reversing.mp4. Guaranteed events, small download.
#   annotated  full 5-minute clips chosen because their .activities.yml declares
#              an activity we want. Better for the long-video case, since a
#              pre-cut example never exercises windowing.
#   raw        the old behaviour, kept on purpose: uneventful clips are what
#              NEGATIVES are made of, and false-positive rate needs them.
#
#   scripts/fetch_meva.sh examples  <out> [limit] [activity-regex]
#   scripts/fetch_meva.sh annotated <out> [limit] [activity-regex]
#   scripts/fetch_meva.sh raw       <out> [limit] [drop]

set -uo pipefail

MODE="${1:-examples}"
OUT="${2:-data/meva}"
LIMIT="${3:-8}"
FILTER="${4:-}"
BUCKET="${MEVA_S3:-s3://mevadata-public-01}"
DROP="${MEVA_DROP:-drops-123-r13}"
MIN_S="${MEVA_MIN_S:-60}"
# Release suffix on video filenames, absent from annotation filenames.
RELEASE="${MEVA_RELEASE:-r13}"

command -v aws >/dev/null 2>&1 || {
  echo "FAIL  aws cli not found. Install it, or pull from your own S3 stage:"
  echo "      make s3-pull S3_BUCKET=your-bucket"
  exit 1
}
mkdir -p "$OUT"

attribution() {
  cat > "${OUT}/ATTRIBUTION.txt" <<'EOF'
MEVA (Multiview Extended Video with Activities)
Kitware Inc., collected for the IARPA Deep Intermodal Video Analytics programme.
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
https://mevadata.org/

Attribution is required for these clips and for any derivative of them,
including trimmed excerpts.
EOF
}

# ---------------------------------------------------------------- examples ---
if [ "$MODE" = "examples" ]; then
  echo "Listing curated activity clips ..."
  listing="${OUT}/.examples"
  aws s3 ls "${BUCKET}/examples/videos/" --no-sign-request > "$listing" 2>/dev/null

  # One clip per DISTINCT activity first, so a small limit spans many event types
  # rather than returning four near-identical takes of the same one.
  awk -v f="$FILTER" '$4 ~ /\.mp4$/ {
      act = $4; sub(/^ex[0-9]+-/, "", act); sub(/\.mp4$/, "", act)
      if (f != "" && act !~ f) next
      if (!(act in seen)) { seen[act] = 1; print act "\t" $4 }
    }' "$listing" | sort > "${OUT}/.manifest"

  count=$(wc -l < "${OUT}/.manifest" | tr -d ' ')
  [ "$count" -eq 0 ] && { echo "FAIL  no example clips matched '${FILTER}'"; exit 1; }

  echo "Fetching up to ${LIMIT} of ${count} distinct activities ..."
  head -n "$LIMIT" "${OUT}/.manifest" > "${OUT}/.take"
  while IFS=$'\t' read -r act name; do
    aws s3 cp "${BUCKET}/examples/videos/${name}" "${OUT}/" \
      --no-sign-request --only-show-errors
    printf "  %-44s %s\n" "$name" "$act"
  done < "${OUT}/.take"
  attribution
  echo
  echo "Each clip is named for the activity it contains."
  echo "CC BY 4.0 — attribution in ${OUT}/ATTRIBUTION.txt"
  exit 0
fi

# --------------------------------------------------------------- annotated ---
if [ "$MODE" = "annotated" ]; then
  echo "Listing annotations to find clips that contain activity ..."
  ann="${OUT}/.annotations"
  aws s3 ls "${BUCKET}/examples/annotations/" --no-sign-request --recursive \
    > "$ann" 2>/dev/null
  awk '/activities\.yml$/ {print $4}' "$ann" > "${OUT}/.annlist"
  [ -s "${OUT}/.annlist" ] || {
    echo "FAIL  no annotation files under ${BUCKET}/examples/annotations/"; exit 1; }

  # The "N instances" meta lines are a cheap index: they declare which activities
  # a clip contains without parsing per-actor detail, so deciding whether a clip
  # is worth a 150 MB download costs a few hundred bytes.
  : > "${OUT}/.manifest"
  n=0
  while read -r key; do
    [ "$n" -ge "$LIMIT" ] && break
    tmp=$(mktemp)
    aws s3 cp "${BUCKET}/${key}" "$tmp" --no-sign-request --only-show-errors 2>/dev/null \
      || { rm -f "$tmp"; continue; }
    acts=$(grep -oE 'meta: "[A-Za-z_]+ [0-9]+ instances"' "$tmp" \
           | sed 's/meta: "//; s/ [0-9]* instances"//' | sort -u | paste -sd, -)
    rm -f "$tmp"
    [ -z "$acts" ] && continue
    if [ -n "$FILTER" ] && ! echo "$acts" | grep -qiE "$FILTER"; then continue; fi
    printf "%s\t%s\n" "$(basename "$key" .activities.yml)" "$acts" >> "${OUT}/.manifest"
    n=$((n + 1))
  done < "${OUT}/.annlist"

  count=$(wc -l < "${OUT}/.manifest" | tr -d ' ')
  [ "$count" -eq 0 ] && { echo "FAIL  no annotated clips matched '${FILTER}'"; exit 1; }

  echo "Fetching ${count} clip(s) that contain annotated activity ..."
  mkdir -p "${OUT}/annotations"
  got=0
  while IFS=$'\t' read -r stem acts; do
    day="${stem%%.*}"
    hour=$(echo "$stem" | cut -d. -f2 | cut -d- -f1)
    # Video filenames carry a RELEASE SUFFIX that the annotation filenames do not:
    #   annotation  ...admin.G329.activities.yml
    #   video       ...admin.G329.r13.avi
    # Building the video name straight from the annotation stem misses it, and
    # every download 404s. Try the suffixed name first, then the bare one.
    for name in "${stem}.${RELEASE}" "${stem}"; do
     for ext in avi mp4; do
      if aws s3 cp "${BUCKET}/${DROP}/${day}/${hour}/${name}.${ext}" "${OUT}/" \
           --no-sign-request --only-show-errors 2>/dev/null; then
        printf "  %-52s %s\n" "${name}.${ext}" "$acts"
        # Keep the annotation beside the clip. It is how we FIND events to
        # hand-label, and how labelling error gets cross-checked afterwards.
        aws s3 cp "${BUCKET}/examples/annotations/${day}/${hour}/${stem}.activities.yml" \
          "${OUT}/annotations/" --no-sign-request --only-show-errors 2>/dev/null
        got=$((got + 1))
        break 2
      fi
     done
    done
  done < "${OUT}/.manifest"

  # Fail loudly. An earlier version swallowed every download error and exited 0
  # having fetched nothing, reporting success for an empty directory.
  if [ "$got" -eq 0 ]; then
    echo
    echo "FAIL  matched ${count} annotated clip(s) but downloaded NONE."
    echo "      The video name is built from the annotation stem plus the release"
    echo "      suffix '${RELEASE}'. Check what the drop actually contains:"
    echo "        aws s3 ls ${BUCKET}/${DROP}/ --no-sign-request | head"
    exit 1
  fi
  attribution
  echo
  echo "Fetched ${got} of ${count} matched clip(s)."
  echo "Annotations kept in ${OUT}/annotations/ — used to LOCATE events."
  echo "The brief requires hand labels, so these cross-check ours, never replace them."
  exit 0
fi

# --------------------------------------------------------------------- raw ---
# Kept deliberately. Uneventful footage is what negatives are made of, and
# false-positive rate cannot be measured without clips where nothing happens.
echo "Listing ${BUCKET}/${DROP} (raw — NOT filtered by activity) ..."
listing="${OUT}/.listing"
if ! aws s3 ls "${BUCKET}/${DROP}/" --no-sign-request --recursive > "$listing" 2>"${OUT}/.err"; then
  echo "FAIL  could not list ${BUCKET}/${DROP}/"; sed 's/^/      /' "${OUT}/.err"; exit 1
fi

# One awk process, no pipe: `... | head -n N` makes head exit early, sends SIGPIPE
# upstream, and under `set -o pipefail` aborts AFTER writing a perfectly good
# manifest — failing with exit 141 that looks like a network error.
awk -v n="$LIMIT" -v min="$MIN_S" '
  $4 ~ /\.(mp4|avi)$/ {
    split($4, p, "/"); name = p[length(p)]
    if (split(name, f, ".") < 3) next
    if (split(f[2], a, "-") != 3 || split(f[3], b, "-") != 3) next
    dur = (b[1]*3600 + b[2]*60 + b[3]) - (a[1]*3600 + a[2]*60 + a[3])
    if (dur < 0) dur += 86400
    if (dur < min) next
    printf "%s\t%d\n", $4, dur
    if (++c >= n) exit
  }' "$listing" > "${OUT}/.manifest"

count=$(wc -l < "${OUT}/.manifest" | tr -d ' ')
[ "$count" -eq 0 ] && { echo "FAIL  no clips of at least ${MIN_S}s found"; exit 1; }

echo "Fetching ${count} clip(s). NOTE: unfiltered by activity — most MEVA footage"
echo "is deliberately uneventful, so expect these to be NEGATIVES."
while IFS=$'\t' read -r key dur; do
  aws s3 cp "${BUCKET}/${key}" "${OUT}/" --no-sign-request --only-show-errors
  printf "  %-52s %4ds\n" "$(basename "$key")" "$dur"
done < "${OUT}/.manifest"
attribution
