##@ Help
# Self-documenting help, generated from the Makefiles themselves.
#
# A target appears here if and only if it carries a `## ` comment, and lands in
# the section marked by the nearest preceding `##@ ` line. That means the
# documented interface cannot drift from the real one — which matters, because
# `make help` is the first thing anyone runs.
#
# Column width is computed from the longest target name rather than fixed, so
# adding a long name can never push a description out of alignment.

.PHONY: help vars

help:  ## [any] show this list
	@printf "\n  \033[1mVideo Reasoning System\033[0m\n"
	@printf "  Find events in a video from a plain-language description.\n\n"
	@awk 'BEGIN {FS = ":.*?## "} \
	  /^##@ / { nm = substr($$0, 5); \
          if (!(nm in idx)) { idx[nm] = ++ns; sect[ns] = nm } \
          cur = idx[nm]; next } \
	  /^[a-zA-Z_-]+:.*?## / { \
	      n++; s[n] = cur; t[n] = $$1; d[n] = $$2; \
	      if (match(d[n], /^\[[a-z]+\] /)) { \
	        g[n] = substr(d[n], 2, RLENGTH - 3); d[n] = substr(d[n], RLENGTH + 1) \
	      } \
	      if (length($$1) > w) w = length($$1); \
	  } \
	  END { \
	    for (i = 1; i <= ns; i++) { \
	      first = 1; \
	      for (j = 1; j <= n; j++) if (s[j] == i) { \
	        if (first) { printf "  \033[1;33m%s\033[0m\n", sect[i]; first = 0 } \
	        col = (g[j] == "gpu") ? 32 : (g[j] == "local") ? 35 : 90; \
	        printf "    \033[36m%-*s\033[0m  \033[%dm%-7s\033[0m %s\n", \
	               w, t[j], col, (g[j] ? "[" g[j] "]" : ""), d[j]; \
	      } \
	      if (!first) printf "\n"; \
	    } \
	  }' $(MAKEFILE_LIST)
	@printf "  \033[1;33mWhere targets run\033[0m\n"
	@printf "    \033[35m[local]\033[0m your machine   \033[32m[gpu]\033[0m the rented GPU box   \033[90m[any]\033[0m either\n\n"
	@printf "  \033[1;33mWorkflow\033[0m\n"
	@printf "    \033[35m1.\033[0m local  make preflight && make build      verify the host, build the image\n"
	@printf "    \033[35m2.\033[0m local  make data                         generate synthetic + fetch real clips\n"
	@printf "    \033[35m3.\033[0m local  make s3-push-eval                  stage ONLY the measured run (~570MB)\n"
	@printf "       \033[2m--- rent the GPU instance, clone this repo on it ---\033[0m\n"
	@printf "    \033[32m4.\033[0m gpu    make preflight-gpu                confirm containers can see the GPU\n"
	@printf "    \033[32m5.\033[0m gpu    make build && make pull           build the finder image, pre-pull the server\n"
	@printf "    \033[32m6.\033[0m gpu    make data-eval                    rebuild the labelled clips (no AWS needed)\n"
	@printf "    \033[32m7.\033[0m gpu    make serve-bg                     start the model, block until ready\n"
	@printf "    \033[32m8.\033[0m gpu    make probe                        CAN the model ground events in time?\n"
	@printf "    \033[32m9.\033[0m gpu    make eval                         the measured run\n"
	@printf "   \033[32m10.\033[0m gpu    make down                         stop containers before you stop paying\n\n"
	@printf "  \033[2mprobe and eval are implemented and exercised against a fake endpoint;\n  no number from either has yet been measured on real hardware.\033[0m\n\n"
	@printf "  \033[2mSteps 5 and 6 are worth doing before you need them: the image and the\n"
	@printf "  weights are multi-GB, and a metered box should not idle on a download --\n  nor on a build. probe, eval and data-eval all run in the finder container,\n  so a fresh clone needs make build first.\033[0m\n\n"
	@printf "  \033[2mmake vars\033[0m   every setting and its current value\n\n"

vars:  ## [any] print every setting, its value and what it controls
	@printf "\n  \033[1m%-13s %-34s %s\033[0m\n" "SETTING" "CURRENT VALUE" "CONTROLS"
	@printf "  %-13s %-34s %s\n" "MODEL"        "$(MODEL)"    "which model is served and queried"
	@printf "  %-13s %-34s %s\n" "BASE_URL"     "$(BASE_URL)" "endpoint; point at a remote GPU box"
	@printf "  %-13s %-34s %s\n" "VLLM_IMAGE"   "$(VLLM_IMAGE)" "pinned server image"
	@printf "  %-13s %-34s %s\n" "VIDEO"        "$(VIDEO)"    "input clip for 'make run'"
	@printf "  %-13s %-34s %s\n" "QUERIES"      "$(QUERIES)"  "event description(s) to look for"
	@printf "  %-13s %-34s %s\n" "DATASET"      "$(DATASET)"  "labelled set for 'make eval'"
	@printf "  %-13s %-34s %s\n" "SAMPLE_FPS"   "$(if $(SAMPLE_FPS),$(SAMPLE_FPS),(config.yaml: 4))" "frames sampled per second"
	@printf "  %-13s %-34s %s\n" "WINDOW_S"     "$(if $(WINDOW_S),$(WINDOW_S),(config.yaml: 12))" "seconds the model sees at once"
	@printf "  %-13s %-34s %s\n" "STRIDE_S"     "$(if $(STRIDE_S),$(STRIDE_S),(config.yaml: 9))" "hop between windows; < window = overlap"
	@printf "  %-13s %-34s %s\n" "OUT_DIR"      "$(OUT_DIR)"  "where results are written"
	@printf "  %-13s %-34s %s\n" "DATA_DIR"     "$(DATA_DIR)" "where clips live"
	@printf "  %-13s %-34s %s\n" "S3_BUCKET"    "$(if $(S3_BUCKET),$(S3_BUCKET),(unset))" "your staging bucket for s3-push/pull"
	@printf "  %-13s %-34s %s\n" "GPU_HOURLY"   "$(if $(GPU_HOURLY),$(GPU_HOURLY),(unset: cost not reported))" "instance \$$/hr, for cost per video-minute"
	@printf "  %-13s %-34s %s\n" "ALLOW_NO_GPU" "$(if $(ALLOW_NO_GPU),$(ALLOW_NO_GPU),(unset: real model required))" "opt in to the stub backend"
	@echo ""
