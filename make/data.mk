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

.PHONY: data-eval meva-index meva-screen meva-shortlist event-sheets meva-plan screen-clips verify-data positive-control prepare-clips frames frames-sweep data data-synthetic data-meva data-supervision data-vantage \
        s3-check s3-push s3-push-dry s3-pull s3-status data-list

# The finder mounts /data READ-ONLY, so the service can never modify a client's
# footage. Data-*producing* targets are the one exception and say so explicitly,
# by overriding that single mount rather than relaxing the default.
DATAGEN_RUN := $(COMPOSE) run --rm datagen

# --- your staging bucket ----------------------------------------------------
# Deliberately EMPTY. The repository is public, and a bucket name is infrastructure
# disclosure rather than a secret — private, credential-gated, but still naming
# someone's account contents to anyone who reads the file.
#
# Set it once in make/local.mk, which is gitignored, and every s3 target works with
# no arguments:
#     echo 'S3_BUCKET = my-bucket' > make/local.mk
# Or per invocation:
#     make s3-push S3_BUCKET=my-bucket
S3_BUCKET  ?=
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

# Selection is by ACTIVITY, not by duration. Most MEVA footage is deliberately
# uneventful — it exists for a challenge about finding rare events in long empty
# streams — so an unfiltered pick returns clips with nothing to label.
#   MEVA_MODE=examples   pre-cut clips named for the activity they contain
#   MEVA_MODE=annotated  full clips whose annotations declare an activity
#   MEVA_MODE=raw        unfiltered; these become NEGATIVES
MEVA_MODE   ?= examples
MEVA_FILTER ?=

data-meva:  ## [local] fetch MEVA clips that CONTAIN events (CC BY 4.0, no credentials)
	@MEVA_MIN_S=$(MEVA_MIN_S) MEVA_DROP=$(MEVA_DROP) bash scripts/fetch_meva.sh \
	  $(MEVA_MODE) $(DATA_DIR)/meva-$(MEVA_MODE) $(MEVA_LIMIT) "$(MEVA_FILTER)"

# Screening clips BEFORE downloading them. MEVA's .geom.yml is 5-70 KB against
# 56-203 MB per video, so the whole corpus screens for less than one clip costs.
MEVA_INDEX  ?= $(DATA_DIR)/meva-index
SHORTLIST   ?= $(MEVA_INDEX)/shortlist.json

data-eval:  ## [local] rebuild the labelled eval set from labels.json (fetches sources)
	@bash scripts/fetch_eval_sources.sh $(DATA_DIR)/eval/labels.json $(DATA_DIR)/meva-annotated
	$(DATAGEN_RUN) python scripts/rebuild_eval_clips.py \
	  --labels /data/eval/labels.json --src /data/meva-annotated --out /data/eval

meva-index:  ## [local] fetch MEVA ANNOTATIONS ONLY (no video) so clips can be screened first
	@bash scripts/fetch_meva_index.sh $(MEVA_INDEX)

meva-screen:  ## [local] rank indexed clips by actor size during their declared events
	$(DATAGEN_RUN) python scripts/screen_geom.py --index /data/meva-index

meva-shortlist:  ## [local] fetch the hand-chosen shortlist (edit shortlist.json first)
	@bash scripts/fetch_meva_shortlist.sh $(SHORTLIST) $(DATA_DIR)/meva-annotated

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

EVAL_DATA_DIR ?= /data/synthetic
CLIP_SRC     ?= /data/meva-annotated
CLIP_SECONDS ?= 120
# Sources to process, not clips in the final eval set — candidates get rejected
# by hand, so this must exceed the 5-10 the brief asks for. Sources are taken in
# sorted order, so a limit below the source count silently drops the LAST ones
# alphabetically, which is where newly fetched clips land.
CLIP_LIMIT   ?= 24

prepare-clips:  ## [local] trim source footage to 1-3 min and scaffold hand-labelling
	$(DATAGEN_RUN) python scripts/prepare_clips.py \
	  --src $(CLIP_SRC) --out /data/eval \
	  --seconds $(CLIP_SECONDS) --limit $(CLIP_LIMIT)
	@echo; echo "Contact sheets: $(DATA_DIR)/eval/sheets/  — label from these"

positive-control:  ## [local] build the ceiling-test set from MEVA example clips
	$(DATAGEN_RUN) python scripts/make_positive_control.py --src /data/meva-examples

SCREEN_DIR ?= /data/meva-annotated

event-sheets:  ## [local] dense sheet per candidate event, for confirming by eye
	$(DATAGEN_RUN) python scripts/event_sheets.py \
	  --labels /data/eval/labels.template.json --data-dir /data/eval \
	  --pad $(SHEET_PAD) --fps $(SHEET_FPS)

meva-plan:  ## [local] turn MEVA annotations into a trim plan + labelling worksheet
	$(DATAGEN_RUN) python scripts/meva_labels.py \
	  --src /data/meva-annotated --seconds $(CLIP_SECONDS)

screen-clips:  ## [any] does anything HAPPEN in these clips? Screen before investing
	$(COMPOSE) run --rm finder python scripts/screen_clips.py $(SCREEN_DIR)

verify-data:  ## [any] check every clip actually shows what its label claims
	$(COMPOSE) run --rm finder python scripts/verify_synthetic.py $(EVAL_DATA_DIR)

data-list:  ## [any] show what is present locally
	@echo "Local data under $(DATA_DIR)/:"
	@find $(DATA_DIR) -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.json' \) 2>/dev/null | sed 's/^/  /' || true
