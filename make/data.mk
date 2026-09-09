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

.PHONY: data data-synthetic data-meva data-supervision data-vantage \
        s3-push s3-pull s3-status data-list

# The finder mounts /data READ-ONLY, so the service can never modify a client's
# footage. Data-*producing* targets are the one exception and say so explicitly,
# by overriding that single mount rather than relaxing the default.
DATAGEN_RUN := $(COMPOSE) run --rm datagen

# --- your staging bucket ----------------------------------------------------
S3_BUCKET ?=
S3_PREFIX ?= video-reasoning/data
AWS_REGION ?= us-east-1

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

s3-push:  ## [local] upload the local data dir to your S3 staging bucket
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	aws s3 sync $(DATA_DIR)/ s3://$(S3_BUCKET)/$(S3_PREFIX)/ --region $(AWS_REGION) --exclude '*.manifest'
	@echo "Staged to s3://$(S3_BUCKET)/$(S3_PREFIX)/"

s3-pull:  ## [gpu] download the staged data from your bucket (run this ON the GPU box)
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	@mkdir -p $(DATA_DIR)
	aws s3 sync s3://$(S3_BUCKET)/$(S3_PREFIX)/ $(DATA_DIR)/ --region $(AWS_REGION)

s3-status:  ## [any] show what is in the staging bucket
	@[ -n "$(S3_BUCKET)" ] || { echo "FAIL  set S3_BUCKET=your-bucket"; exit 1; }
	aws s3 ls s3://$(S3_BUCKET)/$(S3_PREFIX)/ --recursive --human-readable --summarize

data-list:  ## [any] show what is present locally
	@echo "Local data under $(DATA_DIR)/:"
	@find $(DATA_DIR) -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.json' \) 2>/dev/null | sed 's/^/  /' || true
