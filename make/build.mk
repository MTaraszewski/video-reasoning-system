##@ Images
# Image lifecycle. Two images, two jobs:
#   finder  — CPU only, built here: decode, window, merge, schema, CLI, eval
#   vllm    — pinned upstream image, holds the weights and nothing else

.PHONY: build pull shell clean down

build: preflight  ## [any] build the finder image
	$(COMPOSE) build finder

pull:  ## [gpu] pull the pinned vLLM image (large — do this before you need it)
	docker pull $(VLLM_IMAGE)

shell: ## [any] open a shell in the finder container
	$(FINDER_RUN) --entrypoint bash finder

down:  ## [any] stop all containers
	$(COMPOSE) down

clean: down  ## [any] stop containers and drop generated output (keeps the weight cache)
	rm -rf $(OUT_DIR)
	@echo "Removed $(OUT_DIR). The weight cache volume is kept — use 'make clean-weights' to drop it."

clean-weights:  ## [any] drop the model weight cache volume (forces a re-download)
	$(COMPOSE) down -v

##@ Run

.PHONY: run demo plan probe eval sweep viz

# ALLOW_NO_GPU selects the stub backend, so the whole pipeline runs with no model.
# Results are stamped as stub-generated and cannot be scored.
BACKEND ?= auto
_RUN_ENV = ALLOW_NO_GPU=$(ALLOW_NO_GPU)
# NOT $(foreach ...): make splits on whitespace, so a multi-word description
# becomes one query per word — valid JSON for entirely the wrong question.
# The string is passed through whole and split on ";" in Python.

run:  ## [any] find events in YOUR video (VIDEO=... QUERIES="a;b")
	@mkdir -p $(OUT_DIR)
	$(COMPOSE) run --rm -e ALLOW_NO_GPU=$(ALLOW_NO_GPU) finder \
	  video-reasoning run $(VIDEO_IN) --queries "$(QUERIES)" \
	  --backend $(BACKEND) -o /out/events.json \
	  $(if $(SAMPLE_FPS),--fps $(SAMPLE_FPS),) \
	  $(if $(WINDOW_S),--window-s $(WINDOW_S),) \
	  $(if $(STRIDE_S),--stride-s $(STRIDE_S),) \
	  $(if $(PROMPT),--prompt $(PROMPT),) \
	  $(if $(RECORD),--record /out/$(RECORD),)
	@echo; echo "-> $(OUT_DIR)/events.json"

demo:  ## [any] end-to-end on the sample clip, no GPU needed
	@$(MAKE) --no-print-directory run \
	  VIDEO=data/synthetic/box-crossing.mp4 \
	  QUERIES="a red box enters from the left" \
	  BACKEND=stub ALLOW_NO_GPU=1
	@echo; echo "--- events ---"; cat $(OUT_DIR)/events.json

plan:  ## [any] show what a run would cost, without running it
	$(COMPOSE) run --rm finder video-reasoning plan $(VIDEO_IN) --queries "$(QUERIES)" \
	  $(if $(SAMPLE_FPS),--fps $(SAMPLE_FPS),) \
	  $(if $(WINDOW_S),--window-s $(WINDOW_S),) \
	  $(if $(STRIDE_S),--stride-s $(STRIDE_S),)

##@ Planned
# Advertised in the workflow and the docs, so they exist here and say what they
# are waiting on. A missing target prints "No rule to make target", which reads
# as a broken repo rather than an unfinished one.

.PHONY: probe eval eval-control sweep viz

# The probe deliberately sweeps prompt variants: the brief says Edge localises
# "when prompted correctly", so which prompt wins is itself a result.
PROMPTS ?= overlay,native,terse
PROBE_FPS ?= 4

probe:  ## [any] characterise the model: can it ground events, how precisely
	@mkdir -p $(OUT_DIR)
	$(COMPOSE) run --rm -e ALLOW_NO_GPU=$(ALLOW_NO_GPU) finder \
	  video-reasoning probe --backend $(BACKEND) \
	  --prompts $(PROMPTS) --fps $(PROBE_FPS) -o /out/probe.json \
	  $(if $(BASE_URL_OVERRIDE),--base-url $(BASE_URL_OVERRIDE),) \
	  $(if $(RECORD),--record /out/$(RECORD),)
	@echo; echo "-> $(OUT_DIR)/probe.json"

LABELS ?= /data/synthetic/labels.json
EVAL_DATA ?= /data/synthetic

# The positive control lives in its own labels file, never mixed into an eval set.
# queries_for() asks EVERY description of EVERY clip so false positives get
# measured — which means adding control clips to an eval file also adds their
# descriptions to the questions asked of real clips, shifting the FP denominator.
# Containment of the leaked answers works either way; separation keeps the query
# set stable and comparable between runs.
CONTROL_LABELS ?= /data/meva-examples/labels.json
CONTROL_DATA   ?= /data/meva-examples

eval:  ## [gpu] run the labelled set, print the metric table
	@mkdir -p $(OUT_DIR)
	$(COMPOSE) run --rm finder video-reasoning evaluate \
	  --labels $(LABELS) --data-dir $(EVAL_DATA) --backend $(BACKEND) \
	  -o /out/eval.json \
	  $(if $(PROMPT),--prompt $(PROMPT),) \
	  $(if $(SAMPLE_FPS),--fps $(SAMPLE_FPS),) \
	  $(if $(GPU_HOURLY),--gpu-hourly $(GPU_HOURLY),) \
	  $(if $(REPLAY),--replay /out/$(REPLAY),)
	@echo; echo "-> $(OUT_DIR)/eval.json"

eval-control:  ## [gpu] CEILING TEST: can the model find events labelled on-screen?
	@mkdir -p $(OUT_DIR)
	@echo "Ceiling test. These clips carry MEVA's annotations BURNED INTO the"
	@echo "picture, so the model can read the answer. If it fails HERE, it will"
	@echo "fail on clean footage — and the cause is prompting or vision, not"
	@echo "event recognition. Never comparable with eval numbers."
	@echo
	$(COMPOSE) run --rm finder video-reasoning evaluate \
	  --labels $(CONTROL_LABELS) --data-dir $(CONTROL_DATA) \
	  --backend $(BACKEND) -o /out/eval-control.json \
	  $(if $(PROMPT),--prompt $(PROMPT),) \
	  $(if $(SAMPLE_FPS),--fps $(SAMPLE_FPS),) \
	  $(if $(REPLAY),--replay /out/$(REPLAY),)
	@echo; echo "-> $(OUT_DIR)/eval-control.json"

sweep:  ## [gpu] fps / window / stride frontier  (planned - increment 5)
	@echo "not implemented yet: 'sweep' lands in increment 5."
	@exit 1

viz:  ## [any] render an events timeline  (planned - optional)
	@echo "not implemented yet: 'viz' is optional and lands after the eval."
	@exit 1
