##@ Data
# Dataset acquisition and staging.
#
# Two hops on purpose:
#   1. fetch from the original source (slow, once, from anywhere)
#   2. stage into YOUR OWN S3 bucket (fast, repeatable, same-region as the GPU box)
#
# The second hop matters once a GPU is running: pulling clips from a bucket in the
# same region takes seconds and costs nothing, while re-fetching from the original
# sources burns rented GPU minutes on downloads.
#
# LICENCES — see DATASETS.md. Short version:
#   synthetic    ours
#   MEVA         CC BY 4.0, ungated, attribution required
#   supervision  licence UNSTATED — fetch only, never redistribute
#   VANTAGE      evaluation-only, GATED, no redistribution
# Nothing here commits video to the repo. Only our labels are committed.

.PHONY: frames frames-sweep data data-synthetic data-meva data-supervision data-vantage \
        s3-check s3-push s3-push-dry s3-pull s3-status data-list

# The finder mounts /data READ-ONLY, so the service can never modify a client's
# footage. Data-*producing* targets are the one exception and say so explicitly,
# by overriding that single mount rather than relaxing the default.
DATAGEN_RUN := $(COMPOSE) run --rm datagen

# --- your staging bucket ----------------------------------------------------
# Set as the default so s3-push / s3-pull need no arguments. Override per run
# with: make s3-pull S3_BUCKET=other-bucket
S3_BUCKET  ?= mt-video-reasoning-system
S3_PREFIX  ?= data

# Assigned with `=`, NOT `?=`, on purpose. `?=` skips assignment when the name is
# already defined — and make inherits the environment, so an exported
# AWS_REGION (or a profile default of eu-west-1) silently won and pointed every
# transfer at the wrong region. S3 redirects rather than erroring, so the only
# symptom would have been slow, cross-region-billed transfers.
#
# The bucket's region is a fact about the bucket, not a user preference. A plain
# `=` lets the makefile beat the environment while `make ... AWS_REGION=x` on the
# command line still wins. `make s3-check` verifies it against the bucket itself.
AWS_REGION = eu-central-1

# --- MEVA -------------------------------------------------------------------
# Public, no credentials. 470 GB in total, so we take one narrow slice.
MEVA_S3    ?= s3://mevadata-public-01
MEVA_DROP  ?= drops-123-r13
MEVA_LIMIT ?= 12
MEVA_MIN_S ?= 60          # skip MEVA's short fragments; the brief wants 1-3 min clips

data: data-synthetic  ## [local] generate/fetch everything needed for a first run

data-synthetic:  ## [local] generate synthetic clips with EXACT ground truth
	$(DATAGEN_RUN) python scripts/make_synthetic.py --out /data/synthetic

data-meva:  ## [local] fetch a MEVA slice (CC BY 4.0, no credentials needed)
	@MEVA_MIN_S=$(MEVA_MIN_S) bash scripts/fetch_meva.sh $(DATA_DIR)/meva $(MEVA_LIMIT) $(MEVA_DROP)

data-supervision:  ## [local] fetch Roboflow supervision sample videos (licence unstated — do not redistribute)
	$(DATAGEN_RUN) python scripts/fetch_supervision.py --out /data/supervision

data-vantage:  ## [local] fetch VANTAGE-Bench (GATED — needs HF_TOKEN and accepted terms)
	@[ -n "$$HF_TOKEN" ] || { echo "FAIL  VANTAGE-Bench is gated. Accept the terms on Hugging Face,"; \
	  echo "      then: export HF_TOKEN=hf_xxx"; exit 1; }
	$(DATAGEN_RUN) -e HF_TOKEN hf download nvidia/PhysicalAI-VANTAGE-Bench \
	  --repo-type dataset --include 'temporal_localization/*' --local-dir /data/vantage

# --- staging to your own bucket ---------------------------------------------

s3-check:  ## [any] resolve the bucket's real region and check it matches AWS_REGION
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	@bash scripts/s3_check.sh $(S3_BUCKET) $(AWS_REGION)

s3-push: s3-check  ## [local] upload the local data dir to your S3 staging bucket
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	aws s3 sync $(DATA_DIR)/ s3://$(S3_BUCKET)/$(S3_PREFIX)/ \
	  --region $(AWS_REGION) --exclude '*.manifest' --exclude '*.listing*'
	@echo "Staged to s3://$(S3_BUCKET)/$(S3_PREFIX)/"

s3-pull: s3-check  ## [gpu] download the staged data from your bucket (run this ON the GPU box)
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	@mkdir -p $(DATA_DIR)
	aws s3 sync s3://$(S3_BUCKET)/$(S3_PREFIX)/ $(DATA_DIR)/ --region $(AWS_REGION)

s3-status:  ## [any] show what is in the staging bucket
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	@bash scripts/s3_check.sh $(S3_BUCKET) || true
	@echo "prefix: $(S3_PREFIX)/"
	@# `aws s3 ls` exits 1 on an empty prefix. An empty bucket is a normal state,
	@# not a failure, so report it as such rather than propagating the exit code.
	@aws s3 ls s3://$(S3_BUCKET)/$(S3_PREFIX)/ --recursive --human-readable --summarize \
	  --region $(AWS_REGION) || echo "  (empty - nothing staged yet: make s3-push)"

s3-push-dry:  ## [local] show what s3-push WOULD upload, without uploading
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	@aws s3 sync $(DATA_DIR)/ s3://$(S3_BUCKET)/$(S3_PREFIX)/ \
	  --region $(AWS_REGION) --exclude '*.manifest' --exclude '*.listing*' --dryrun

frames:  ## [any] dump sampled frames with burned-in timestamps, to inspect legibility
	@mkdir -p $(OUT_DIR)/frames
	$(COMPOSE) run --rm finder video-reasoning frames $(VIDEO_IN) \
	  -o /out/frames $(if $(SAMPLE_FPS),--fps $(SAMPLE_FPS),) $(if $(LIMIT),--limit $(LIMIT),)
	@echo "-> $(OUT_DIR)/frames"

frames-sweep:  ## [any] ONE frame at several overlay font scales, to compare legibility
	@mkdir -p $(OUT_DIR)/sweep
	$(COMPOSE) run --rm finder video-reasoning frames $(VIDEO_IN) -o /out/sweep --sweep
	@echo "-> $(OUT_DIR)/sweep  (same frame, varying font size — compare legibility)"

data-list:  ## [any] show what is present locally
	@echo "Local data under $(DATA_DIR)/:"
	@find $(DATA_DIR) -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.json' \) 2>/dev/null | sed 's/^/  /' || true
