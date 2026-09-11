# Every knob, in one place. Override any of these on the command line:
#
#   make run VIDEO=/path/clip.mp4 QUERIES="a forklift reverses"
#   make serve MODEL=nvidia/Cosmos-Reason2-8B

# --- model / serving --------------------------------------------------------
MODEL      ?= nvidia/Cosmos3-Edge
BASE_URL   ?= http://vllm:8000/v1
# 0.29.0, not the 0.21.0 that NVIDIA's Cosmos3-Nano recipe implies for CUDA 13.
# 0.21.0 fails on Cosmos3-Edge before it reaches the GPU:
#   The checkpoint you are trying to load has model type `cosmos3_edge`
#   but Transformers does not recognize this architecture.
# The bundled Transformers predates the model. 0.29.0 resolves the architecture as
# Cosmos3EdgeForConditionalGeneration and loads. Eight minor versions between the
# documented recipe and one that works — measured, not inferred.
VLLM_IMAGE ?= vllm/vllm-openai:v0.29.0
VLLM_PORT  ?= 8000

# --- inputs -----------------------------------------------------------------
VIDEO   ?= data/synthetic/box-crossing.mp4
# Same path as seen inside the container, where ./data is mounted at /data.
VIDEO_IN = $(patsubst ./data/%,/data/%,$(patsubst data/%,/data/%,$(VIDEO)))
QUERIES ?= a red box enters from the left
DATASET ?= data/synthetic/labels.json

# --- pipeline knobs ---------------------------------------------------------
# Left unset here on purpose: defaults live in config.yaml with their rationale,
# and are only overridden when explicitly passed.
SAMPLE_FPS ?=
WINDOW_S   ?=
STRIDE_S   ?=

# --- paths ------------------------------------------------------------------
OUT_DIR  ?= ./out
DATA_DIR ?= ./data

# --- cost reporting ---------------------------------------------------------
# Deliberately unset. Unset means cost is NOT reported, rather than reported wrong.
GPU_HOURLY ?=

# --- escape hatch -----------------------------------------------------------
# Opt in to the stub backend so the pipeline runs with no GPU. Results are
# stamped as stub-generated and the eval harness refuses to score them.
ALLOW_NO_GPU ?=

# --- derived ----------------------------------------------------------------
COMPOSE     := docker compose
FINDER_RUN  := $(COMPOSE) run --rm finder

export MODEL BASE_URL VLLM_IMAGE VLLM_PORT OUT_DIR DATA_DIR GPU_HOURLY ALLOW_NO_GPU

# Event-sheet rendering. Padding is lead-in/lead-out around a candidate window,
# so the eye can see the action START rather than only see it in progress.
SHEET_PAD ?= 4.0
SHEET_FPS ?= 2.0
