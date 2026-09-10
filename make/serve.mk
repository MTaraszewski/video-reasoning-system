##@ Serving
# Model server lifecycle.
#
# We serve the Cosmos 3 REASONER only — video in, text out. Stock vLLM does this
# natively via Cosmos3EdgeForConditionalGeneration. The `--omni` flag and the
# vLLM-Omni project exist to serve the diffusion GENERATOR, which we never use.
#
# Version pin: vllm 0.21.0 on CUDA 13 drivers, 0.19.1 on CUDA 12.8.
# Source: https://recipes.vllm.ai/nvidia/Cosmos3-Nano

.PHONY: serve serve-bg serve-wait serve-logs serve-down models

# Confirmed against a real Cosmos3-Edge serve, not copied from a recipe.
#
# --media-io-kwargs '{"video": {"num_frames": -1}}' was carried over from NVIDIA's
# Cosmos3-Nano recipe and REMOVED for two reasons. First, it broke: the JSON is
# double-quoted inside an already double-quoted make variable, so the shell strips
# the inner quotes and vLLM receives `{video: {num_frames: -1}}`, rejecting it with
#     argument --media-io-kwargs: Value {video: {num_frames: -1}} cannot be converted
# Second, and the reason it is not worth re-quoting: that flag controls how vLLM
# decodes a VIDEO FILE it is given. We never use that path — frames are decoded
# here, timestamped, and sent as images — so the flag configures a code path this
# system does not exercise.
VLLM_ARGS ?= --max-model-len 32768 --allowed-local-media-path /data
VLLM_EXTRA ?=

serve: preflight-gpu  ## [gpu] start the model server on the local GPU (foreground)
	MODEL="$(MODEL)" VLLM_ARGS="$(VLLM_ARGS) $(VLLM_EXTRA)" $(COMPOSE) up vllm

serve-bg: preflight-gpu  ## [gpu] start the model server in the background
	MODEL="$(MODEL)" VLLM_ARGS="$(VLLM_ARGS) $(VLLM_EXTRA)" $(COMPOSE) up -d vllm
	@$(MAKE) --no-print-directory serve-wait

serve-wait:  ## [gpu] block until the endpoint answers and confirm which model it serves
	@bash scripts/wait_for_endpoint.sh $(VLLM_PORT) "$(MODEL)"

serve-logs:  ## [gpu] tail the model server logs
	$(COMPOSE) logs -f vllm

serve-down:  ## [gpu] stop the model server
	$(COMPOSE) stop vllm

models:  ## [any] ask the endpoint what it is actually serving
	@curl -s http://localhost:$(VLLM_PORT)/v1/models | python3 -m json.tool

##@ Testing without a GPU

.PHONY: fake-serve fake-down fake-test

SCENARIO ?= think

fake-serve:  ## [any] start a fake vLLM endpoint (SCENARIO=think|mixed|truncated|...)
	SCENARIO=$(SCENARIO) $(COMPOSE) up -d fake-vllm
	@bash scripts/wait_for_endpoint.sh $(VLLM_PORT) "$(MODEL)"

fake-down:  ## [any] stop the fake endpoint
	$(COMPOSE) stop fake-vllm && $(COMPOSE) rm -f fake-vllm

# SCENARIOS overrides which cases run. Each is "name:min:max" expected events,
# so the harness can actually fail rather than just report numbers.
#   make fake-test
#   make fake-test SCENARIOS="hallucinate:0:0"
#   make fake-test SCENARIOS="think:1:9 empty:0:0"
SCENARIOS ?=

fake-test:  ## [any] run the REAL vllm backend against a fake endpoint, all scenarios
	@bash scripts/fake_test.sh $(SCENARIOS)
