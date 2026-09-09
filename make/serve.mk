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

# Flags carried over from NVIDIA's published reasoner recipe. Edge has no recipe
# of its own yet, so these are inferred and must be confirmed on first serve.
VLLM_ARGS ?= --max-model-len 32768 \
             --media-io-kwargs '{"video": {"num_frames": -1}}' \
             --allowed-local-media-path /data
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
