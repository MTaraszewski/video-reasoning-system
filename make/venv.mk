##@ Development
# Local development environment, via uv.
#
# The container is the supported way to run this — nothing needs to be installed
# on the host to use the service. These targets exist for editing and debugging
# with a working language server, and resolve from the SAME uv.lock the image
# uses, so a local venv and the container never drift.

.PHONY: venv lock sync

venv: sync  ## [local] create the local .venv from uv.lock

sync:  ## [local] install locked dependencies into .venv
	@command -v uv >/dev/null 2>&1 || { \
	  echo "FAIL  uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync --frozen
	@echo
	@echo "Activate with:  source .venv/bin/activate"

lock:  ## [local] re-resolve dependencies and update uv.lock (after editing pyproject.toml)
	uv lock
	@echo "uv.lock updated — rebuild the image so the container matches: make build"
