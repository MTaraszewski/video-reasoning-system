##@ Images
# Image lifecycle. Two images, two jobs:
#   finder  — CPU only, built here: decode, window, merge, schema, CLI, eval
#   vllm    — pinned upstream image, holds the weights and nothing else

.PHONY: transitions offset-test build pull shell clean down

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

# STRATEGY picks the engine: `windows` asks the model when the event happened
# (Approach 1), `states` captions and derives the event from state transitions
# (Approach 3). Both stay runnable so a comparison is one flag apart.
STRATEGY   ?=
STATES_MAP ?= /app/states.json
TRIGGER    ?=

run:  ## [any] find events in YOUR video (VIDEO=... QUERIES="a;b" STRATEGY=states)
	@mkdir -p $(OUT_DIR)
	$(COMPOSE) run --rm -e ALLOW_NO_GPU=$(ALLOW_NO_GPU) finder \
	  video-reasoning run $(VIDEO_IN) --queries "$(QUERIES)" \
	  --backend $(BACKEND) -o /out/events.json \
	  $(if $(SAMPLE_FPS),--fps $(SAMPLE_FPS),) \
	  $(if $(WINDOW_S),--window-s $(WINDOW_S),) \
	  $(if $(STRIDE_S),--stride-s $(STRIDE_S),) \
	  $(if $(PROMPT),--prompt $(PROMPT),) \
	  $(if $(STRATEGY),--strategy $(STRATEGY) --states-map $(STATES_MAP),) \
	  $(if $(TRIGGER),--trigger,) \
	  $(if $(SHARED),--shared-caption,) \
	  $(if $(STEP_S),--step-s $(STEP_S),) \
	  $(if $(SPAN_S),--span-s $(SPAN_S),) \
	  $(if $(RECORD),--record /out/$(RECORD),)
	@echo; echo "-> $(OUT_DIR)/events.json"

demo:  ## [any] end-to-end on the sample clip, no GPU needed
	@# Generate the clip if it is missing. `data/` is gitignored -- it holds GBs of
	@# footage -- so on a fresh clone this file does not exist, and a demo that
	@# fails on first run is the one failure the brief is explicit about.
	@[ -f $(DATA_DIR)/synthetic/box-crossing.mp4 ] \
	  || $(MAKE) --no-print-directory data-synthetic
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
PROBE_OUT ?= /out/probe.json
# The probe defaults to synthetic clips because their ground truth is exact by
# construction. Point it at the hand-labelled set to ask whether the precision
# floor measured on synthetic stimuli survives on real footage -- and set
# PROBE_EXCERPT so both are sampled at a comparable rate.
PROBE_LABELS  ?= /data/synthetic/labels.json
PROBE_DATA    ?= /data/synthetic
PROBE_EXCERPT ?=

probe:  ## [any] characterise the model: can it ground events, how precisely
	@mkdir -p $(OUT_DIR)
	@# The probe measures against synthetic clips whose ground truth is exact by
	@# construction. Same reasoning as demo: generate them if absent.
	@[ -f $(DATA_DIR)/synthetic/labels.json ] \
	  || $(MAKE) --no-print-directory data-synthetic
	$(COMPOSE) run --rm -e ALLOW_NO_GPU=$(ALLOW_NO_GPU) finder \
	  video-reasoning probe --backend $(BACKEND) \
	  --prompts $(PROMPTS) --fps $(PROBE_FPS) -o $(PROBE_OUT) \
	  --labels $(PROBE_LABELS) --data-dir $(PROBE_DATA) \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(PROBE_EXCERPT),--excerpt-s $(PROBE_EXCERPT),) \
	  $(if $(BASE_URL_OVERRIDE),--base-url $(BASE_URL_OVERRIDE),) \
	  $(if $(RECORD),--record /out/$(RECORD),)
	@echo; echo "-> $(PROBE_OUT)"

# The hand-labelled real set is what `make eval` should measure -- it is the
# deliverable. Pointing this at /data/synthetic silently evaluated the wrong
# dataset and the run looked entirely normal while doing it.
# Two-stage extraction and subsampling. DETECT=1 asks a yes/no presence question
# before localising and takes confidence from its logprob; EVAL_LIMIT runs only
# the first N clips, so a change can be tried for pennies before the full
# cross-product is committed to a metered box.
EVAL_OUT   ?= /out/eval.json
DETECT     ?=
EVAL_LIMIT ?=
DETECT_THRESHOLD ?=

LABELS ?= /data/eval/labels.json
EVAL_DATA ?= /data/eval

# The positive control lives in its own labels file, never mixed into an eval set.
# queries_for() asks EVERY description of EVERY clip so false positives get
# measured — which means adding control clips to an eval file also adds their
# descriptions to the questions asked of real clips, shifting the FP denominator.
# Containment of the leaked answers works either way; separation keeps the query
# set stable and comparable between runs.
TRANS_STEP ?= 1.0
TRANS_SPAN ?= 2.0

OFFSET_VIDEO ?= /data/eval/2018-03-07.16-50-01.16-55-01.admin.G326.r13.mp4
OFFSET_QUERY ?= a person opens a building door
OFFSET_TRUTH ?= 3.0 5.733
OFFSETS      ?= 0,3,9,15
# `localize` asserts the event is present. That is the point here: under a prompt
# that permits refusal the model declines on most windows, including ones that
# DO contain the event, so there is nothing to measure a slide against.
OFFSET_PROMPT ?= localize

CONTROL_LABELS ?= /data/meva-examples/labels.json
CONTROL_DATA   ?= /data/meva-examples

# Where is a running eval right now? The eval prints one line per CLIP and
# nothing in between, so a states run goes quiet for 30+ minutes at a time and
# looks hung. This reconstructs the position from the model server's request log
# without touching the running job -- see scripts/eval_progress.sh for why that
# is sound.
.PHONY: eval-progress eval-watch
eval-progress:  ## [any] where is a running eval? one-shot
	@SUBJECTS=$(SUBJECTS) POLLS=$(POLLS) CLIPS=$(CLIPS) STEP_S=$(STEP_S) \
	  scripts/eval_progress.sh

eval-watch:  ## [any] same, refreshing every WATCH_S seconds (default 30)
	@SUBJECTS=$(SUBJECTS) POLLS=$(POLLS) CLIPS=$(CLIPS) STEP_S=$(STEP_S) \
	  scripts/eval_progress.sh $(WATCH_S)

# Defaults describe the run we measure most; override for a different shape.
SUBJECTS ?= 4
POLLS    ?= 119
CLIPS    ?= 4
STEP_S   ?= 1.0
WATCH_S  ?= 30

.PHONY: validate
validate:  ## [any] check a results file against the contract (FILE=out/events.json)
	$(COMPOSE) run --rm finder video-reasoning validate /out/$(notdir $(FILE))

FILE ?= out/events.json

.PHONY: schema
schema:  ## [any] regenerate schema.json, the machine-readable output contract
	@# Captured on the HOST, not written inside the container: the repo root is
	@# not mounted, so -o /app/schema.json would write into a layer that is
	@# discarded when the container exits.
	@$(COMPOSE) run --rm --no-TTY finder video-reasoning schema > schema.json
	@echo "-> schema.json  ($$(wc -l < schema.json) lines)"

# Every check that needs no GPU, in one command. This is what answers "does the
# repo work on my machine" before anyone rents an instance -- it exercises the
# host, the image, the whole pipeline end to end, the data, the decoder, and the
# real model adapter against a fake endpoint.
#
# It does NOT prove the model works. Nothing here touches a GPU or real weights,
# and `demo` runs on the stub, whose results the eval harness refuses to score.
SMOKE ?= preflight build demo verify-data frames fake-test

.PHONY: smoke
smoke:  ## [any] run every check that needs no GPU, and report which passed
	@fail=0; \
	for t in $(SMOKE); do \
	  printf '  %-14s ' "$$t"; \
	  if $(MAKE) --no-print-directory $$t >/tmp/smoke-$$t.log 2>&1; then \
	    echo "PASS"; \
	  else \
	    echo "FAIL   -> /tmp/smoke-$$t.log"; fail=1; \
	  fi; \
	done; \
	$(MAKE) --no-print-directory fake-down >/dev/null 2>&1 || true; \
	echo; \
	if [ $$fail -eq 0 ]; then \
	  echo "all $(words $(SMOKE)) checks passed - no GPU was used"; \
	  echo "next: make pull && make serve-bg && make serve-wait && make eval"; \
	else \
	  echo "SOME CHECKS FAILED - see the logs named above"; exit 1; \
	fi

eval:  ## [gpu] run the labelled set, print the metric table
	@mkdir -p $(OUT_DIR)
	@# The labelled clips are rebuilt from labels.json, which IS committed. Without
	@# this a fresh clone fails here with "video not found" and no hint that one
	@# command fixes it.
	@[ -f $(DATA_DIR)/eval/labels.json ] || { echo "FAIL  no labels at $(DATA_DIR)/eval/labels.json"; exit 1; }
	@$(MAKE) --no-print-directory data-eval
	$(COMPOSE) run --rm finder video-reasoning evaluate \
	  --labels $(LABELS) --data-dir $(EVAL_DATA) --backend $(BACKEND) \
	  -o $(EVAL_OUT) \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(DETECT),--detect,) \
	  $(if $(DETECT_THRESHOLD),--detect-threshold $(DETECT_THRESHOLD),) \
	  $(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT),) \
	  $(if $(PROMPT),--prompt $(PROMPT),) \
	  $(if $(SAMPLE_FPS),--fps $(SAMPLE_FPS),) \
	  $(if $(GPU_HOURLY),--gpu-hourly $(GPU_HOURLY),) \
	  $(if $(STRATEGY),--strategy $(STRATEGY) --states-map $(STATES_MAP),) \
	  $(if $(SHARED),--shared-caption,) \
	  $(if $(STEP_S),--step-s $(STEP_S),) \
	  $(if $(SPAN_S),--span-s $(SPAN_S),) \
	  $(if $(TRIGGER),--trigger,) \
	  $(if $(REPLAY),--replay /out/$(REPLAY),)
	@echo; echo "-> $(OUT_DIR)$(patsubst /out%,%,$(EVAL_OUT))"

transitions:  ## [gpu] score caption-parse-derive across every labelled event
	$(COMPOSE) run --rm finder python scripts/run_transitions.py \
	  --labels $(LABELS) --data-dir $(EVAL_DATA) \
	  --step $(TRANS_STEP) --span $(TRANS_SPAN) \
	  $(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT),) \
	  --out /out/transitions.json

offset-test:  ## [gpu] does the reported time follow the EVENT or the WINDOW?
	$(COMPOSE) run --rm finder python scripts/window_offset_test.py \
	  --video $(OFFSET_VIDEO) --query "$(OFFSET_QUERY)" \
	  --truth $(OFFSET_TRUTH) --offsets $(OFFSETS) --backend $(BACKEND) \
	  --prompt $(OFFSET_PROMPT)

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
	  $(if $(MODEL),--model $(MODEL),) \
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
