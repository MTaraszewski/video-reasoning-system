#!/usr/bin/env bash
# Run the same pipeline over several models, one at a time, on one GPU.
#
# The comparison this exists for is the last row of models.tsv: Cosmos-Reason2-8B
# was post-trained from Qwen3-VL-8B-Instruct. Same architecture, same parameter
# count; the only difference is NVIDIA's physical-AI training. Running both under
# an identical harness isolates what that training buys for temporal
# localisation -- a question we have not found answered anywhere public.
#
# Serial, never parallel: two models on one card would contend for VRAM and make
# every latency and cost number a measurement of the contention instead of the
# model.
#
#   bash scripts/model_sweep.sh                 # every model in models.tsv
#   bash scripts/model_sweep.sh --only Qwen     # substring filter
#   bash scripts/model_sweep.sh --skip-eval     # probe only, for a cheap first pass
set -euo pipefail

REG="${MODELS_TSV:-models.tsv}"
OUT="${OUT_DIR:-./out}"
ONLY=""; SKIP_EVAL=""; SKIP_PROBE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --skip-probe) SKIP_PROBE=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

[ -f "$REG" ] || { echo "FAIL  no model registry at ${REG}" >&2; exit 1; }
mkdir -p "$OUT"

slug() { echo "$1" | tr '/' '-' | tr '[:upper:]' '[:lower:]'; }

started=$(date +%s)
declare -a DONE=() SKIPPED=()

# Skip comments, the header, and blanks. Read tab-separated so a role containing
# spaces stays in one field.
# Ask the card what it has, so a model that cannot fit is skipped before its
# weights are downloaded rather than after. Qwen3-VL-8B cost 138 seconds of
# download and three minutes of load before vLLM refused to start.
VRAM_GIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
           | head -1 | awk '{printf "%d", $1/1024}')
[ -n "$VRAM_GIB" ] && echo "GPU has ${VRAM_GIB} GiB" || VRAM_GIB=0

while IFS=$'\t' read -r id gated maxlen minvram role; do
  case "$id" in ''|'#'*|'id') continue ;; esac
  [ -n "$ONLY" ] && case "$id" in *"$ONLY"*) ;; *) continue ;; esac

  echo
  echo "=============================================================="
  echo "  $id"
  echo "  $role"
  echo "=============================================================="

  # A gated model without a token fails several minutes in, after the image is
  # up and the download starts. Check before spending that time.
  if [ "$gated" = "yes" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "SKIP  gated, and HF_TOKEN is not set."
    echo "      Accept the terms on https://huggingface.co/${id} then:"
    echo "        export HF_TOKEN=hf_xxx"
    SKIPPED+=("$id (gated, no HF_TOKEN)")
    continue
  fi

  # A model that cannot fit is a skip with a reason, not a five-minute failure.
  if [ "$VRAM_GIB" -gt 0 ] && [ -n "$minvram" ] && [ "$VRAM_GIB" -lt "$minvram" ]; then
    echo "SKIP  needs ${minvram} GiB, this card has ${VRAM_GIB} GiB."
    echo "      Running it anyway would mean fewer frames and a shorter context"
    echo "      than the 4B baseline got, which breaks the comparison."
    SKIPPED+=("$id (needs ${minvram} GiB, have ${VRAM_GIB})")
    continue
  fi

  s=$(slug "$id")

  # One model at a time on one card.
  make --no-print-directory serve-down >/dev/null 2>&1 || true
  if ! MODEL="$id" VLLM_ARGS="--max-model-len ${maxlen} --allowed-local-media-path /data" \
       make --no-print-directory serve-bg; then
    echo "FAIL  ${id} did not come up -- see: make serve-logs"
    SKIPPED+=("$id (did not serve)")
    continue
  fi

  ok=1
  if [ -z "$SKIP_PROBE" ]; then
    echo "-- probe (synthetic, exact ground truth) --"
    MODEL="$id" make --no-print-directory probe \
      PROMPTS=overlay PROBE_OUT="/out/probe-${s}.json" || { echo "  PROBE FAILED"; ok=0; }
  fi

  if [ -z "$SKIP_EVAL" ]; then
    echo "-- eval (hand-labelled clips, full cross-product) --"
    MODEL="$id" make --no-print-directory eval \
      PROMPT=overlay EVAL_OUT="/out/eval-${s}.json" || { echo "  EVAL FAILED"; ok=0; }
  fi

  # "swept 1 model" while the probe errored is a lie the summary told once
  # already. A model counts as run only if what we asked for actually ran.
  if [ "$ok" = "1" ]; then DONE+=("$id"); else SKIPPED+=("$id (ran, but a stage failed)"); fi
done < "$REG"

make --no-print-directory serve-down >/dev/null 2>&1 || true

echo
echo "swept ${#DONE[@]} model(s) in $(( ($(date +%s) - started) / 60 )) min"
for m in "${DONE[@]:-}"; do [ -n "$m" ] && echo "  ran     $m"; done
for m in "${SKIPPED[@]:-}"; do [ -n "$m" ] && echo "  skipped $m"; done
echo
echo "compare them:  make compare"
