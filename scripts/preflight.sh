#!/usr/bin/env bash
# Environment checks. Each fails hard, reports what it found, and says what to do.
#
# Lives here rather than in the Makefile deliberately: macOS ships GNU Make 3.81,
# which silently ignores .ONESHELL, so multi-line make recipes are not portable.
# Anything with real logic belongs outside the Makefile anyway.
#
#   scripts/preflight.sh host
#   scripts/preflight.sh gpu

set -uo pipefail

CUDA_PROBE_IMAGE="${CUDA_PROBE_IMAGE:-nvidia/cuda:12.4.0-base-ubuntu22.04}"

check_host() {
  local fail=0

  if command -v docker >/dev/null 2>&1; then
    echo "ok    docker $(docker --version | awk '{print $3}' | tr -d ,)"
  else
    echo "FAIL  docker not found."
    echo "      Install Docker, or on macOS OrbStack / Docker Desktop."
    fail=1
  fi

  if docker compose version >/dev/null 2>&1; then
    echo "ok    compose $(docker compose version --short 2>/dev/null)"
  else
    echo "FAIL  'docker compose' not available — the v2 plugin is required."
    fail=1
  fi

  if docker info >/dev/null 2>&1; then
    echo "ok    docker daemon reachable"
  else
    echo "FAIL  docker daemon not reachable. Is it running?"
    fail=1
  fi

  return $fail
}

check_gpu() {
  echo "Probing GPU access from inside a container..."

  # This single command is worth more than checking nvidia-smi on the host,
  # because it proves three things at once: a GPU exists, the driver works, and
  # the NVIDIA container toolkit is installed. A host-only check passes happily
  # while containers still see no GPU — the most common silent failure.
  if docker run --rm --gpus all "$CUDA_PROBE_IMAGE" nvidia-smi >/dev/null 2>&1; then
    docker run --rm --gpus all "$CUDA_PROBE_IMAGE" \
      nvidia-smi --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader | sed 's/^/ok    /'
    return 0
  fi

  cat <<'EOF'
FAIL  no GPU available to containers.

      This one check covers three separate failures:
        - no NVIDIA GPU present
        - driver not installed or not working
        - nvidia-container-toolkit missing (usually this one — host
          nvidia-smi passes while containers still see nothing)

      To run without a GPU, opt in explicitly:
        make demo ALLOW_NO_GPU=1

      Stub results are stamped as such and cannot produce a metric.
EOF
  return 1
}

case "${1:-host}" in
  host) check_host ;;
  gpu)  check_host && check_gpu ;;
  *)    echo "usage: $0 [host|gpu]" >&2; exit 2 ;;
esac
