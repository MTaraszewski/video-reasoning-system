##@ Setup
# Environment checks.
#
# The split that matters: only the MODEL SERVER needs a GPU. The finder decodes
# video, plans windows and merges results — all CPU work. So `serve` gates on a
# local GPU, while client targets only need a reachable endpoint. Gating
# everything on a local GPU would break running the CLI against a remote box,
# which is exactly what BASE_URL exists for.
#
# The logic lives in scripts/preflight.sh: macOS ships GNU Make 3.81, which
# silently ignores .ONESHELL, so multi-line recipes are not portable.

.PHONY: preflight preflight-gpu

preflight:  ## [any] check the host can run this at all
	@bash scripts/preflight.sh host

preflight-gpu:  ## [gpu] check a GPU is visible TO CONTAINERS (required before serve)
	@bash scripts/preflight.sh gpu
