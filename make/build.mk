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

##@ Planned
# These are advertised in the workflow and in the docs, so they exist here and say
# what they are waiting on. A missing target prints "No rule to make target", which
# reads as a broken repo rather than an unfinished one.

.PHONY: run demo probe eval sweep viz

run:  ## [any] find events in YOUR video  (planned - increment 2g)
	@echo "not implemented yet: 'run' lands in increment 2g (CLI wiring)."
	@echo "The pipeline exists - see src/video_reasoning/core.py find_events()."
	@exit 1

demo:  ## [any] end-to-end on the sample clip  (planned - increment 2g)
	@echo "not implemented yet: 'demo' lands in increment 2g (CLI wiring)."
	@exit 1

probe:  ## [gpu] characterise the model  (planned - increment 3)
	@echo "not implemented yet: 'probe' lands in increment 3, on the GPU box."
	@exit 1

eval:  ## [gpu] run the labelled set, print the metric table  (planned - increment 4)
	@echo "not implemented yet: 'eval' lands in increment 4 (metrics + labels)."
	@exit 1

sweep:  ## [gpu] fps / window / stride frontier  (planned - increment 5)
	@echo "not implemented yet: 'sweep' lands in increment 5."
	@exit 1

viz:  ## [any] render an events timeline  (planned - optional)
	@echo "not implemented yet: 'viz' is optional and lands after the eval."
	@exit 1
