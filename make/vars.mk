# Every knob, in one place. Override any of these on the command line:
#
#   make run VIDEO=/path/clip.mp4 QUERIES="a forklift reverses"
#   make serve MODEL=nvidia/Cosmos-Reason2-8B

# --- model / serving --------------------------------------------------------
MODEL      ?= nvidia/Cosmos3-Edge
BASE_URL   ?= http://vllm:8000/v1
VLLM_IMAGE ?= vllm/vllm-openai:v0.21.0
VLLM_PORT  ?= 8000

# --- inputs -----------------------------------------------------------------
VIDEO   ?= data/synthetic/box-crossing.mp4
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
