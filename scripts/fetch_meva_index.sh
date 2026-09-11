#!/usr/bin/env bash
# Fetch MEVA's ANNOTATIONS ONLY — no video — so clips can be screened before any
# large download.
#
# Why this exists: the first selection pass chose clips because their annotation
# declared an activity, and three of six turned out to show that activity at
# 38-121 px, far below what a human can label. The annotation that answers "is
# the actor big enough?" is `.geom.yml`, and it was being discarded by an awk
# filter that kept only `.activities.yml`.
#
# Annotations are 5-70 KB against 56-203 MB per video, so screening the whole
# corpus costs less than downloading one clip.
#
#   bash scripts/fetch_meva_index.sh data/meva-index
set -euo pipefail

OUT="${1:-data/meva-index}"
BUCKET="${MEVA_S3:-s3://mevadata-public-01}"
PREFIX="${BUCKET}/examples/annotations"
AWS=(aws s3 --no-sign-request --no-cli-pager)

mkdir -p "$OUT"

echo "Listing ${PREFIX} ..."
# One listing, then filter locally. Piping a long listing into head would send
# SIGPIPE to the aws process and abort the run after writing a partial manifest,
# which is how an earlier version reported success having fetched nothing.
"${AWS[@]}" ls --recursive "${PREFIX}/" > "${OUT}/.listing"

n_act=$(grep -c 'activities\.yml$' "${OUT}/.listing" || true)
n_geom=$(grep -c 'geom\.yml$' "${OUT}/.listing" || true)
echo "  ${n_act} activity file(s), ${n_geom} geometry file(s)"

if [ "$n_geom" -eq 0 ]; then
  echo "FAIL  no .geom.yml found under ${PREFIX}" >&2
  exit 1
fi

# Pull both kinds in one sync rather than per-file cp: ~130 round trips become one.
echo "Syncing annotations to ${OUT} ..."
"${AWS[@]}" sync "${PREFIX}/" "${OUT}/" \
  --exclude "*" --include "*.activities.yml" --include "*.geom.yml" --quiet

got=$(find "$OUT" -name '*.geom.yml' | wc -l | tr -d ' ')
echo "OK    ${got} clip(s) indexed under ${OUT}"
[ "$got" -gt 0 ] || { echo "FAIL  sync wrote nothing" >&2; exit 1; }
