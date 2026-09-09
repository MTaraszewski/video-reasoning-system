#!/usr/bin/env bash
# Fetch a narrow slice of MEVA.
#
# MEVA is CC BY 4.0 — Kitware, collected for the IARPA DIVA programme — and is
# hosted free via the AWS Public Dataset Program. No credentials, no agreement,
# hence --no-sign-request. ATTRIBUTION IS REQUIRED wherever these clips or any
# derivative of them are used.
#
# The full corpus is ~470 GB across 4001 clips, so we take a handful.
#
#   scripts/fetch_meva.sh <out_dir> [limit] [drop]

set -euo pipefail

OUT="${1:-data/meva}"
LIMIT="${2:-12}"
DROP="${3:-drops-123-r13}"
# Minimum clip length in seconds. The brief wants 1-3 minute clips, and MEVA
# mixes 5-minute recordings with short fragments of the same scene, so without a
# floor an alphabetical pick returns mostly 15-25 second offcuts.
MIN_S="${MEVA_MIN_S:-60}"
BUCKET="${MEVA_S3:-s3://mevadata-public-01}"

command -v aws >/dev/null 2>&1 || {
  echo "FAIL  aws cli not found. Install it, or fetch from your own S3 stage instead:"
  echo "      make s3-pull S3_BUCKET=your-bucket"
  exit 1
}

mkdir -p "$OUT"
manifest="${OUT}/.manifest"

echo "Listing ${BUCKET}/${DROP} ..."

# Deliberately NOT piped straight into `head`. `head` closes the pipe as soon as
# it has enough lines, aws keeps writing, and the resulting SIGPIPE surfaces as
# "Broken pipe" / exit 141 — which looks like a network failure but is not.
# Listing to a file first keeps the two stages independent.
listing="${OUT}/.listing"
if ! aws s3 ls "${BUCKET}/${DROP}/" --no-sign-request --recursive > "$listing" 2>"${OUT}/.listing.err"; then
  echo "FAIL  could not list ${BUCKET}/${DROP}/"
  sed 's/^/      /' "${OUT}/.listing.err"
  echo "      Available drops:"
  aws s3 ls "${BUCKET}/" --no-sign-request | sed 's/^/        /'
  exit 1
fi

# Select by DURATION, not by name. MEVA filenames encode it:
#   2018-03-05.09-49-37.09-50-00.school.G474.r13.avi
#                ^start    ^end                       -> 23 s
# The corpus interleaves 5-minute recordings with short fragments of the same
# scene, so taking the first N alphabetically yields mostly 15-25 second offcuts
# — useless for a benchmark about long-video handling.
#
# One awk process, no pipe: `... | head -n N` would make head exit early, send
# SIGPIPE upstream, and with `set -o pipefail` abort the script AFTER writing a
# perfectly good manifest, failing with exit 141 that looks like a network error.
awk -v n="$LIMIT" -v min="$MIN_S" '
  $4 ~ /\.(mp4|avi)$/ {
    key = $4
    m = split(key, parts, "/")
    name = parts[m]
    if (split(name, f, ".") < 3) next
    if (split(f[2], a, "-") != 3 || split(f[3], b, "-") != 3) next
    start = a[1]*3600 + a[2]*60 + a[3]
    end   = b[1]*3600 + b[2]*60 + b[3]
    dur   = end - start
    if (dur < 0) dur += 86400          # crossed midnight
    if (dur < min) next
    printf "%s\t%d\n", key, dur
    if (++c >= n) exit
  }' "$listing" > "$manifest"

count=$(wc -l < "$manifest" | tr -d ' ')
if [ "$count" -eq 0 ]; then
  echo "FAIL  no clips of at least ${MIN_S}s found under ${BUCKET}/${DROP}/"
  echo "      Lower the floor:  MEVA_MIN_S=30 make data-meva"
  echo "      Or list the drops: aws s3 ls ${BUCKET}/ --no-sign-request"
  exit 1
fi

echo "Fetching ${count} clip(s) of at least ${MIN_S}s into ${OUT} ..."
while IFS=$'\t' read -r key dur; do
  aws s3 cp "${BUCKET}/${key}" "${OUT}/" --no-sign-request --only-show-errors
  printf "  %-52s %4ds\n" "$(basename "$key")" "$dur"
done < "$manifest"

cat > "${OUT}/ATTRIBUTION.txt" <<'EOF'
MEVA (Multiview Extended Video with Activities)
Kitware Inc., collected for the IARPA Deep Intermodal Video Analytics programme.
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
https://mevadata.org/

Attribution is required for these clips and for any derivative of them,
including trimmed excerpts.
EOF

echo
echo "Done. CC BY 4.0 — attribution recorded in ${OUT}/ATTRIBUTION.txt"
