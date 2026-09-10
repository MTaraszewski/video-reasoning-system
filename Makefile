# Video Reasoning System — entry point.
#
# This file stays thin on purpose: it sets the shell, then includes one .mk per
# concern from make/. Anything with real logic in it belongs in Python, where it
# can be debugged — the Makefile is a readable layer over `docker compose`.
#
#   make help    to see everything

SHELL := /bin/bash
.DEFAULT_GOAL := help

# NOTE: no .ONESHELL here. It needs GNU Make >= 3.82 and macOS ships 3.81, where
# it is silently ignored — multi-line recipes would then break only on a Mac.
# Anything needing real shell logic lives in scripts/ instead.

# Include order sets the order sections appear in `make help`.
include make/vars.mk
include make/preflight.mk
include make/build.mk
include make/data.mk
include make/serve.mk
include make/venv.mk
include make/help.mk

# Personal settings that must not be committed: bucket names, a preferred region,
# GPU_HOURLY for cost reporting. Gitignored, and optional - the leading dash means
# make does not complain when it is absent, so a fresh clone works unchanged.
-include make/local.mk
