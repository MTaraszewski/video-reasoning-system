# Plan & working log — video event-finding service

Living document. Tracks decisions, verified facts, open questions and step status.
Rule for this file: **nothing enters as fact without a primary source.** Anything
unverified is marked `UNVERIFIED` and must not leak into README/DESIGN.

---

## 0. The assignment, distilled

Source: [`reference/take-home.md`](reference/take-home.md)

Build a service: *(video file, one or more plain-language event descriptions)* →
list of events with **start time, end time, matched description, confidence /
ranking signal**, in a **strict, documented JSON schema**.

### Hard requirements

| # | Requirement |
|---|---|
| R1 | Long-video handling: frame sampling, windowing, cross-boundary merge/dedup, reporting partially-seen events |
| R2 | **Document those decisions** — explicitly stated to be part of the task |
| R3 | Model: NVIDIA **Cosmos 3 Edge** recommended. Alternative allowed if justified, and still: open weights, video input, temporal localisation, a context limit worked around |
| R4 | Hosted commercial API = explicit fallback only, never the primary engine |
| R5 | **Runs on the reviewer's machine in a few minutes, first try, without reading the code** — "the part we are strict about" |
| R6 | Own eval set: **5–10 clips, 1–3 min each, hand-labelled start/end**, rights-clean. Report **a temporal metric you can defend** |
| R7 | Report where the model's limits are |
| R8 | Deliverable = GitHub repo |
| R9 | ~1 week |

### Explicitly NOT evaluated
Tests, CI, linters, production-grade deployment, service scaling, advanced
security. Focus = *solution, value for customers, ease of use*.

### How the assignment was framed (from the covering email)

> "we generally use one of two strategies: *build with Roboflow* (an open-ended
> project) or **"build around capability"** (where we ask candidates to build a
> solution centered on a specific model or group of models that **currently pose a
> challenge for tool builders**). Given the recent trend toward multi-modality and
> video understanding, I've decided to go with the latter. It's a real-world
> challenge with **a few tricky edges**"

This is the most useful context we have, and it sets the priorities:

- **The model is expected to be difficult.** "Poses a challenge for tool builders"
  is not a warning to work around — it is the subject of the exercise. Roboflow
  builds tools; they are asking whether we can productise something that resists
  productisation, and what we learn doing it.
- **"A few tricky edges" implies the edges are findable and known to them.** They
  will recognise which ones we hit and which we missed. Handling an edge visibly and
  documenting it beats a clean happy path.
- **Reinforces §"claim under test".** Difficulty is the premise. A submission
  reporting that everything worked smoothly has probably not looked hard enough.
- **"a specific model or group of models"** — comparing against a second model is
  within the spirit, and is the cheapest way to make "where are the limits"
  interpretable. Kept as optional and secondary to characterising the primary.
- The offer to answer questions is repeated here, having also appeared in the brief.
  Owner decision stands: none asked.

**Tricky edges identified so far.** Tracked deliberately — each is either handled
visibly or documented as a known limit:

| Edge | Where it bites |
|---|---|
| Video exceeds the model's effective view | Windowing, overlap, merge. Named in the brief |
| Localisation mechanism undocumented for Edge | The capability probe |
| Reasoning model output is not bare JSON | Response parsing in the adapter |
| Two towers, and "omni" is not what we need | Serving. Easy to waste days on vLLM-Omni |
| Repo id is `Cosmos3-Edge`, not `Cosmos-3-Edge` | Nothing runs if guessed |
| No published VRAM figure, no Edge serving recipe | Instance sizing, first-run reliability |
| 4B at robot-control resolution 640x360 | Small burned text may not survive downscaling |
| Confidence from a zero-shot VLM is uncalibrated | The ranking signal must not be oversold |
| Sampling rate sets a hard floor on boundary precision | Interpreting tIoU at 0.7 |
| Fixed-camera domain weaknesses Roboflow already published | Fast motion, small similar objects |

### Standing offer from the brief
"Questions before you start are welcome; ask them." — an open channel to Paweł,
not yet used.

---

## 1. Verified facts (Step 0)

Checked 2026-09-09. Each line carries its source.

### Cosmos 3 Edge — EXISTS, and is our primary engine
- Repo ID is **`nvidia/Cosmos3-Edge`** (also `nvidia/Cosmos3-Edge-Policy-DROID`).
  Note: **not** `nvidia/Cosmos-3-Edge`, which is what the earlier draft used.
  — <https://huggingface.co/blog/nvidia/cosmos3edge>
- 4B parameters. Released at SIGGRAPH, 2026-07-20. Weights + code + post-training
  recipes published.
  — <https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai>
- **Licence: OpenMDW 1.1. Not gated** — no HF token, no licence click-through.
  This is a real R5 advantage over Cosmos Reason 2, which *is* gated.
  — <https://huggingface.co/nvidia/Cosmos3-Edge>
- Omnimodal: text, image, video, audio, action in; robot actions out at 15 Hz on
  Jetson Thor. Autoregressive tower (vision+language) + diffusion tower
  (vision/audio/action). Robot-control resolution 640x360.
- **Video input recommended at 4 fps for reasoning tasks.** Reasoner supports
  long-context input **up to 256K tokens**. **BF16 only** — FP4/FP8/FP16 are not
  officially supported.
- Hardware tested: H100 80GB, B200 192GB, and Jetson 16-128GB.
- Ships with a companion 2B dense reasoning module (Nemotron-powered), runnable on
  Jetson Orin 8GB.
- Ranked #1 open model on VANTAGE-Bench, Arena Bench, PAI-Bench and R-Bench.

### Serving Cosmos 3 Edge — stock vLLM, no vLLM-Omni needed
Cosmos 3 splits into two towers, and **only one of them is ours**:

| Tower | What it does | Served by |
|---|---|---|
| **Reasoner** (autoregressive) | video/image/text in -> text out | **stock vLLM**, TensorRT-LLM |
| **Generator** (diffusion) | generates video/image/audio/action | vLLM-Omni only |

We need the **Reasoner only**. Video in, timestamped events out. Nothing is
generated. So the `--omni` flag, the `vllm/vllm-omni:cosmos3` image and the
`--no-guardrails` flag are all **out of scope** — they belong to the generator path.

- `nvidia/Cosmos3-Edge` is listed in vLLM's supported-models table:
  architecture **`Cosmos3EdgeForConditionalGeneration`**, described as
  "Cosmos3-Edge (understanding tower)", inputs **`T + I^E+ + V^E+`**
  (text + multiple images + multiple videos), LoRA supported.
  — <https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md>
- Implementation lives in `vllm.model_executor.models.cosmos3_edge`:
  `Cosmos3EdgeForConditionalGeneration` = **Nemotron-H backbone + SigLIP2 vision
  encoder** + patch merger/projector, with interleaved multimodal RoPE, registered
  via `@MULTIMODAL_REGISTRY.register_processor()` and implementing both
  `_process_image_input()` and `_process_video_input()`.
  — <https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/cosmos3_edge/>
- Note Edge is a **separate integration** from Nano/Super, which use the unified
  `Cosmos3ForConditionalGeneration` / Omni checkpoint.
- Version pinning, from the Cosmos3 reasoner recipe:
  **`vllm==0.21.0` on CUDA 13 drivers, `vllm==0.19.1` on CUDA 12.8.**
  — <https://recipes.vllm.ai/nvidia/Cosmos3-Nano>
- Flags worth carrying over from that reasoner recipe:
  `--media-io-kwargs '{"video": {"num_frames": -1}}'`, `--allowed-local-media-path`,
  `--async-scheduling`, `--mm-encoder-tp-mode data`.
- `UNVERIFIED` / to settle on the box:
  - No published **Cosmos3-Edge** recipe or VRAM figure. 4B in BF16 is ~8 GB of
    weights plus KV cache and the vision tower, so a 24 GB card is the working
    hypothesis — **must be measured**, and it decides the AWS instance.
  - Nano's recipe needs `--hf-overrides '{"architectures": [...]}'` to select the
    reasoner out of its unified checkpoint. Edge's checkpoint should map to its own
    architecture directly and not need this. **Confirm on first serve.**
  - Exact vLLM version that first shipped `cosmos3_edge`.

### Cosmos 3 architecture — two towers, and we use one

Cosmos 3 is a **Mixture-of-Transformers (MoT)**: *one* model containing *two*
transformer towers.
— <https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/>

| Tower | Type | Does | Modalities |
|---|---|---|---|
| **Reasoner** | autoregressive | Interprets input, "understands motion, object interactions, and other physical context". The "brain" | text / image / video / audio / action **in**, text **out** |
| **Generator** | diffusion | Produces future observations and action sequences by iterative denoising, conditioned on the reasoner | generates video, image, audio, action |

The decisive sentence for us: **"The reasoner operates independently, but
generation requires both towers working together."** Our task is video in →
timestamped text out. That is the Reasoner alone. We never generate anything.

**What "omni" means.** *Omnimodal* = the union of modalities across both towers —
text, image, video, audio and robot actions, in and out, in one model. **vLLM-Omni**
is a separate project extending vLLM to serve **diffusion-based generation**; the
`--omni` flag switches that on. Since we never generate, `--omni`, the
`vllm/vllm-omni:cosmos3` image and `--no-guardrails` are all **out of scope**.
Stock vLLM serving the understanding tower is the whole requirement.

**Family sizes** (note the discrepancy, resolve before quoting any of them):
- **Edge — 4B**, edge/Jetson class. — HF launch blog.
- **Nano — 16B**, workstation class (RTX PRO 6000). — NVIDIA developer blog.
- **Super — 64B**, datacenter (Hopper/Blackwell). — NVIDIA developer blog.
- `CONFLICT`: Roboflow's blog describes the variant it tested as "Cosmos 3 Super
  (32B)". NVIDIA's developer blog says Super is 64B. Do not quote a Super size
  until this is resolved.
- That developer blog predates Edge and does not mention it at all.

### Cosmos3-Edge and temporal localisation — VERIFIED ABSENT
Checked the model card directly. **Temporal localization, timestamps, timestamp
overlays on frames, event detection, start/end times, VANTAGE-Bench and dense
captioning are not mentioned anywhere on it.**
— <https://huggingface.co/nvidia/Cosmos3-Edge>

Its stated intended uses are: *multimodal understanding, world simulation, future
prediction, action reasoning and Physical AI applications*; robotics manipulation
and control; autonomous-vehicle action prediction; image-to-video generation;
text-to-image generation; action-trajectory generation. The Reasoner is documented
as accepting text, text+image or text+video and producing text, with up to 256K
context — but **no temporal-grounding capability is claimed.**

This is a documented absence, not evidence the capability is missing. It means we
cannot cite a source for it and **must establish it empirically.**

### Why Edge might behave differently from Cosmos Reason 2
The two models share almost nothing below the API:

| | Cosmos3-Edge | Cosmos-Reason2-8B |
|---|---|---|
| Language backbone | **Nemotron-H** | **Qwen3-VL-8B-Instruct** |
| Vision encoder | **SigLIP2** + patch merger/projector | Qwen3-VL's own |
| Size | 4B | 8B |
| Timestamp localisation | not documented | **documented on the model card** |
| Gated | no | yes |

- Backbone details come from vLLM's implementation:
  `Cosmos3EdgeAttention` is "Nemotron-H attention with interleaved multimodal
  RoPE"; `Cosmos3EdgeTextModel` is a "Nemotron-H backbone"; `Cosmos3EdgeVisionModel`
  is the "complete Cosmos vision tower" over a "vLLM packed SigLIP2" encoder.
  — <https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/cosmos3_edge/>
- **Nemotron-H is a hybrid Mamba-2 / Transformer architecture**: most self-attention
  layers are replaced by Mamba state-space layers with constant compute and memory
  per token, giving up to ~3x faster inference.
  — <https://arxiv.org/abs/2504.03624>

Reading a burned-in timestamp requires two things to hold at once: the vision
encoder must **resolve small text** at the sampled resolution, and the model must
**bind that text to the frame's position** in the sequence. Neither transfers
automatically from a Qwen3-VL-based model to a Nemotron-H + SigLIP2 one.

A concrete additional worry: Edge is documented at **robot-control resolution
640x360**. A timestamp legible at 1080p may not survive downscaling to that, and
overlay font size is a parameter we control — so **overlay legibility is itself a
variable to sweep**, not a fixed choice.

### VANTAGE-Bench — matches the task, but **cannot be our eval set**
- Four pillars: **Semantic** (event verification, video QA), **Spatial** (referring
  expressions, pointing, object localisation), **Temporal** (**Temporal
  Localization**, dense video captioning), **Spatio-Temporal** (single object
  tracking). Domains: warehouse, transportation, smart spaces — fixed-camera.
  Temporal Localization is scored by **mIoU**, with **Precision@0.5** secondary.
  — <https://github.com/Clemson-Capstone/VANTAGE-Bench>
- **Ruled out as a source of eval clips (R6).** Licence is
  **`nvidia-evaluation-data-license`**, the dataset is **gated and "for evaluation
  purposes only"**, and **ground truth is deliberately withheld** — scoring happens
  server-side via a submission portal. We need clips we can hand-label and ship in a
  public repo; this permits neither.
  — <https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench>
  - `CONFLICT`: the GitHub README implies videos are downloadable from HF while the
    dataset card says they are not. Either way the licence blocks redistribution, so
    the conclusion stands.
- Still useful as **prior art for task framing**: it confirms mIoU-style temporal
  overlap is the accepted metric for exactly this task, which supports our choice
  in the design note.

### VANTAGE-Bench leaderboard — what it does and does not tell us
Public zero-shot leaderboard, live since 2026-05-27.
— <https://vantage-bench.org/> · <https://huggingface.co/spaces/clemson-computing/VANTAGE-Bench-Leaderboard>

| Rank | Model | Overall |
|---|---|---|
| 3 | Cosmos3-Super 64B | 63.01 |
| 5 | Cosmos3-Nano 16B | 60.67 |
| 7 | Cosmos-Reason2-8B | 54.18 |

- **`nvidia/Cosmos3-Edge` does not appear on the public leaderboard.** NVIDIA's
  claim that Edge "ranks #1 on VANTAGE-Bench for vision analytics" among 4B models
  is therefore **self-reported and not independently verifiable**. Do not cite it as
  third-party evidence.
- Per-task `Temp Loc` values were not retrievable from the rendered pages; the
  column exists but the numbers need the live leaderboard. `UNVERIFIED`.
- Two things worth noting from the overall column: scores sit in the **54-63 out of
  100** band, so this task is far from solved even for the best models; and the
  ordering is **size-monotonic** within the Cosmos family, which is not encouraging
  for a 4B model.
- Resolves the earlier size conflict: the leaderboard lists **Super at 64B**, matching
  NVIDIA's developer blog and contradicting the "Super (32B)" description elsewhere.

### The assessor's claim about Cosmos 3 Edge — a hypothesis under test, not evidence
The brief states, verbatim:

> "We recommend building on **NVIDIA Cosmos 3 Edge**. It takes video input, reasons
> about what happens over time, and **can localise events with timestamps when
> prompted correctly**"

The tempting reading is "the assessor says it works, so it works." That reading is
a trap, and the brief says so itself in its opening paragraph:

> "take a capable model with a new behaviour, **understand what it actually
> guarantees**, and turn it into something a client can call"

Plus: *"we are interested to find out where the **limits** of their abilities are"*,
and *"we care about your decisions and how you communicate them at least as much as
about the code."* Three separate lines instructing us to **establish** the model's
behaviour rather than inherit it. A submission that assumes the claim and builds on
top of it has skipped the part being graded.

**So the claim is treated as a hypothesis with a stated test**, not as a citation:

| | |
|---|---|
| Claim | Edge can localise events with timestamps when prompted correctly |
| Source | The assignment brief. Not NVIDIA's model card, which is silent on this |
| Status | **Untested** |
| Test | The capability probe, Step 3 — measured against clips with exact known ground truth |
| Reported | Either way. A negative result is a finding the brief explicitly asks for |

Note also what the brief does **not** say: it does not say Edge localises by reading
timestamps **burned into frames**. That mechanism is documented for Cosmos Reason 2
only. For Edge the mechanism is unstated, so the probe tests both burned overlays
and vLLM's native video-timing path, and the qualifier *when prompted correctly*
makes prompt design part of the experiment rather than a detail settled afterwards.

**What the probe buys us regardless of outcome.** It measures the model's temporal
grounding floor — the best precision achievable before any windowing or merging is
layered on. Every later metric is interpreted against that number. Without it we
cannot tell our pipeline's error apart from the model's.

### Roboflow's own published view of Cosmos 3
Directly relevant: the company setting this assignment has published its own
evaluation. — <https://blog.roboflow.com/cosmos-3-vision/> (Erik Kokalj, Developer
Experience @ Roboflow, 2026-06-03)
- Tested **Cosmos 3 Super (32B) in thinking mode** only; Nano and Edge untested.
- Scenarios: airport gate cargo movement, warehouse loading dock, kitchen assembly
  line — all fixed-camera, all state-segmentation framings.
- **Strengths:** reliably segments activity into structured state sequences with no
  fine-tuning; good on slow-changing states (e.g. pallet fill level).
- **Limitations:** struggles with **fast-moving actions** and **small, similar
  objects**; 45/67 on Visual Understanding Evals, below Qwen 3.5 27B; weak on
  spatial understanding and object counting; failed reliably on the overhead
  kitchen line.
- **Implication for R7:** our eval set should deliberately probe these known weak
  spots. Confirming or contradicting a Roboflow-published finding with our own
  measured numbers is the strongest possible answer to "where are the limits".

### Cosmos Reason 2 — EXISTS; our documented fallback backend
- Repo IDs: `nvidia/Cosmos-Reason2-2B`, `nvidia/Cosmos-Reason2-8B`,
  `nvidia/Cosmos-Reason2-32B`. — <https://huggingface.co/nvidia/Cosmos-Reason2-8B>
- 8B: gated (contact info required), **NVIDIA Open Model License, commercial use
  permitted**, base model **Qwen3-VL-8B-Instruct**, BF16, **256K input tokens**.
- **Minimum 32 GB GPU memory** for 8B; 24 GB for 2B. Tested on H100 / A100;
  supported microarchitectures Hopper and Blackwell.
- Video input: mp4. **Recommended `fps=4` to match the training setup.**
  `max_tokens` 4096+ to avoid truncated responses.

### The timestamp-overlay crux — CONFIRMED
- Official model card: *"Our AI model recognizes timestamps added at the bottom of
  each frame for accurate temporal localization."*
  — <https://huggingface.co/nvidia/Cosmos-Reason2-8B>
- The same phrasing appears in Cosmos 3 material. So the earlier draft's central
  design premise is **correct**, not a guess.
- **BUT `UNVERIFIED` for Cosmos3-Edge specifically:** the Cosmos3-Edge model card
  makes **no mention of timestamp burning or temporal localisation**. Edge's
  reasoning comes from a Nemotron-based module, not from Cosmos-Reason2. Whether
  Edge reads burned-in timestamps is an **empirical question we must test first
  thing on the GPU box.** This is the single largest technical risk in the build.
- NVIDIA's temporal-localization recipe adds timestamps via
  `add_timestamps_to_all_videos_adaptive.py`; tested fps ∈ {4, 8, 12} and found
  **8 fps optimal for temporal localization specifically**; success criterion
  **mean relative error < 30 %** of subtask duration.
  — <https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason1/temporal_localization/post_training.html>
- Caveat: that <30 % figure comes from a **post-trained** (fine-tuned on MimicGen)
  model, not a zero-shot guarantee. Must not be quoted as a zero-shot number.

### Serving the fallback (Cosmos Reason 2)
- Cosmos Reason 2 supports Transformers and vLLM; **`vllm>=0.11.0` recommended**.
  Documented serving command:
  ```
  vllm serve <model> --max-model-len 8192 --gpu-memory-utilization 0.8 \
    --reasoning-parser qwen3 --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --enable-prefix-caching --port 8010
  ```
  — <https://github.com/nvidia-cosmos/cosmos-reason2>
- **`--reasoning-parser qwen3` means the model emits a `<think>` block before its
  answer.** Any output parsing must account for this.

### Local environment (verified by running)
Python 3.14.6 · `uv` · `docker` (OrbStack) · `ffmpeg` · `git` present.
**No `gh` CLI.** No NVIDIA GPU on this Mac.

---

## 2. Engineering constraints we must get right

Derived from the verified facts above. These are the things that silently produce
wrong-but-plausible results, so each is called out before it is written.

**Model interface**
- Cosmos 3 is a **reasoning model**: it emits a think/answer structure, not bare
  JSON. Output parsing must extract the answer span first. Naively constraining the
  whole response to a JSON schema and parsing it wholesale will fail.
- Serve via the pinned **`vllm/vllm-omni:cosmos3`** image with `--omni`; record the
  exact digest. Never rely on a floating `latest` tag for a first-try run.
- **BF16 only.** No fp8 quantisation, despite the flag existing.
- Timestamps are burned bottom-of-frame; **absolute, video-relative** time so a
  reported value needs no per-window remapping.

**Sampling & windowing**
- 4 fps is the documented input rate for reasoning. 8 fps is what NVIDIA's
  temporal-localisation recipe found optimal. Treat fps as a **measured
  accuracy/cost knob**, and report the sweep rather than asserting a default.
- Clamp every model-reported timestamp into the window's real span. A window must
  never emit an event outside the footage it actually saw.

**Merging across windows**
- Chained merging must be **bounded**. An unbounded "merge if the gap to the
  running span is small" rule collapses a dense candidate stream into one giant
  event, because the running end keeps advancing.
- Confidence combination must not saturate. If agreement across windows only ever
  pushes confidence up, everything ends at ~1.0 and the ranking signal the brief
  asks for stops discriminating.

**Evaluation**
- Metrics must be computed **per video, then aggregated**. Pooling every
  prediction and every label into flat lists and matching on the description
  string lets a prediction from clip A satisfy a label in clip B — and query
  strings repeat across clips by design. This inflates every number reported.
- No metric may be reported from a mock/offline path. Mock mode exists to prove
  the pipeline runs, never to produce numbers.

**Documentation discipline**
- Every knob documented as a decision must actually be read by the code.
- Every example output in the README must match what the schema really emits.
- No hardware, cost or accuracy figure appears in a doc until it has been measured
  on the box or cited to a primary source.

## 3. Decisions taken

Moved to **[`DECISIONS.md`](DECISIONS.md)** — every choice with its reason, the
alternatives rejected and why, what would change it, and the reversals kept on the
record.

Summary of what is settled: `nvidia/Cosmos3-Edge` primary with
`nvidia/Cosmos-Reason2-8B` as reference and `Qwen/Qwen3-VL-8B-Instruct` as a
controlled baseline · stock vLLM, not vLLM-Omni · everything in Docker · CLI
primary · model behind a pluggable adapter · capability probe as a first-class
component · six dataset sources selected by failure axis · the Cosmos 3 paper
benchmarks as the one reproduction target.

---

## 4. Open questions

- [ ] **By what mechanism does Cosmos3-Edge localise events in time?** The brief
      asserts it can "localise events with timestamps when prompted correctly", so
      the capability is not in question — the *mechanism* is. Test both on the box:
      (a) timestamps burned into frames, as documented for Cosmos Reason 2, and
      (b) vLLM's native video input path supplying frame timing. Pick on evidence.
      Prompt design is a first-class experiment here, given the brief's qualifier.
- [ ] **Actual VRAM for the Edge reasoner.** Decides the AWS instance and the
      cost-per-video-minute figure. Working hypothesis: fits 24 GB.
- [ ] Which vLLM version first shipped `cosmos3_edge`; pin it exactly.
- [ ] Does Edge need `--hf-overrides` to select the reasoner architecture?
- [x] ~~VANTAGE-Bench as an eval source~~ — **closed, ruled out.** Gated,
      evaluation-only licence, ground truth withheld server-side. Need another
      source of real footage.

- [ ] Optimal fps for Edge: 4 (documented input rate) vs 8 (best in NVIDIA's
      temporal-localisation recipe, on a different model). Measure, don't assume.
- [ ] Do we compare against a second model at all, or spend the GPU hours on
      fps/window sweeps for the primary? The covering email's "a specific model
      **or group of models**" makes a comparison defensible, and a lone tIoU number
      is hard to interpret without one.
- [ ] **Capability matrix not yet written down.** Three tables proposed and agreed
      in principle: (A) serving preconditions, (B) the brief's three claims per
      model, (C) localisation decomposed into emits / parseable / in-range /
      accurate, with a **stub control row** so a `PASS` proves the model rather than
      the harness. Decision pending: hand-write into this file now, or have
      `make probe` emit them as JSON and render the markdown so they cannot drift.
- [x] ~~Rights-clean real footage for R6~~ — **sources chosen**, see
      [`DATASETS.md`](DATASETS.md). Three tiers: synthetic (ours, exact ground
      truth), **MEVA** (CC BY 4.0, no login, fixed-camera, includes
      `person_opens_facility_door`), and Roboflow's `supervision` assets
      (fetch-only, licence unstated). Remaining work is tracked in that file.
- [ ] **Nothing in the eval set has forklifts**, one of the brief's own examples.
      Needs a separate hunt if warehouse footage matters.
- [ ] **Ask Roboflow about the `supervision` sample-video licence** — unstated, and
      the answer may simply be "they are ours, use them".

---

## 5. Tooling — skills

Skills live in `.claude/skills/` **inside this repo and are committed**. Each is
created when the work that needs it arrives, not before, so it is shaped by real
use rather than guesswork.

| Skill | Job | Trigger | Status |
|---|---|---|---|
| **`model-facts`** | Verify any model / framework / dataset / licence claim against primary sources before it enters a doc or a decision. Owns the status vocabulary (`?` / `D` / `B` / measured / `CONFLICT`), the source hierarchy, and the rules that documented-absence is not absence and a self-reported ranking is not third-party evidence | Step 0 | **Created** |
| `gpu-runbook` | Provision, serve and tear down the rented GPU box. Capture **measured** VRAM, latency and cost, so no guessed number reaches a document | Step 3 | Deferred |
| `cv-eval` | Temporal metrics, the labelling format, clip sourcing and licence checks, per-video metric isolation | Step 4 | Deferred |
| `first-run-check` | Simulate the reviewer: clean machine, follow the README literally, read no code, timebox to minutes. Guards the brief's strictest bar | Before Step 7 | Deferred |

---

## 6. Experiment plan

What gets run on the rented AWS GPU, against what, and what each run is for.
Dataset detail lives in [`DATASETS.md`](DATASETS.md).

### 6.1 Models

R3 requires open weights, video input, temporal localisation, and a context limit
to work around. All four are served by **stock vLLM** on the same box, one at a
time, so only `MODEL` changes between runs.

| Model | Size | Role | Temporal localisation | Access |
|---|---|---|---|---|
| **`nvidia/Cosmos3-Edge`** | 4B | **Primary** — the brief's recommendation | **`B`** — asserted by the brief, absent from the model card. *This is what we test* | OpenMDW 1.1, **ungated** |
| **`nvidia/Cosmos-Reason2-8B`** | 8B | **Reference** — its model card documents the timestamp mechanism, so it proves the approach works independently of Edge | **`D`** | NVIDIA Open Model License, **gated** |
| `nvidia/Cosmos-Reason2-2B` | 2B | Size-scaling point, 24 GB | `D` family | gated |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | **Controlled baseline** — see below | not claimed | open |

**Why the Qwen baseline earns its GPU hours.** Cosmos-Reason2-8B's base model *is*
Qwen3-VL-8B-Instruct. Same architecture, same parameter count; the only difference
is NVIDIA's physical-AI post-training. Running both isolates that post-training and
answers a question we have not found published anywhere: **does it actually buy
anything for temporal localisation, or would the base model do?** That is a
controlled experiment rather than a leaderboard, and it costs one extra model load.

**On "a context limit you have to work around".** A 256K context appears to
dissolve the problem. It does not. At 4 fps a 3-minute clip is ~720 frames, and
each frame costs hundreds to thousands of vision tokens — so the binding limit is
**frames per call**, not tokens of text. The exact figure per model is `UNVERIFIED`
and is measured on the box; it is what sets window length, and therefore the whole
long-video strategy.

### 6.2 Reproduction targets

Two candidates. They validate different things and only one is a priority.

| Source | Published numbers | Reproducible locally | Validates | Priority |
|---|---|---|---|---|
| **Cosmos 3 technical report** — arXiv 2606.02800 | **Cosmos3-Edge**: CVBench 84.9 · MMBench-Dev 76.6 · RealWorldQA 73.3 · VideoPhy2 40.3 | **Yes** — public ground truth | **That we deployed Edge correctly** | **Do it** |
| NVIDIA temporal-localisation recipe | Zero-shot mean relative error 52.0 / 61.3 / 68.45 % on MimicGen | Yes, but the recipe is **Cosmos-Reason1** (Qwen-VL based) on 7-second clips | Prompt design; an external control on the harness | Harvest the prompt; full rerun optional |

**Why the first one matters more than it looks.** It catches the failure most
likely to pass unnoticed: a serving bug. Wrong preprocessing, wrong frame handling
or wrong precision produces a model that runs, answers plausibly, and scores badly
— and we would spend the week concluding "Edge cannot localise events" when the
real fault was ours. Hitting ~76.6 on MMBench-Dev says the deployment is sound
before we trust a single temporal number.

**Why the second is demoted.** Its two benefits are separable and mostly available
for free. The prompt can simply be read and adapted — reproducing the numbers is
not a prerequisite for using it. Harness validation is already covered by synthetic
clips with exact constructed ground truth. What remains is that the control would
be *external* rather than self-made, which is worth something but not days, on
7-second robot-manipulation footage, against a one-week deadline.

**Not a reproduction target: VANTAGE-Bench.** Its ground truth is withheld and
scoring is server-side, so its leaderboard numbers cannot be reproduced locally by
any means. It stays in as a dataset we hand-label ourselves. Its published scores —
Cosmos3-Super 63.01, Cosmos3-Nano 60.67, Cosmos-Reason2-8B 54.18 — are context,
not a target.

### 6.3 The run matrix

The same hand-labelled clips run against every model, so **only clip count drives
cost**; adding a model is one more pass, adding a dataset is free.

```
                 synthetic   MEVA   VANTAGE   supervision   ComplexVAD/  Assembly101
                                                             StreetScene
Cosmos3-Edge         .        .        .           .             .            .
Cosmos-Reason2-8B    .        .        .           .             .            .
Cosmos-Reason2-2B    .        .        .           .             .            .
Qwen3-VL-8B          .        .        .           .             .            .
```

Reported per cell: tIoU metrics (`R@1@{0.3,0.5,0.7}`, mean tIoU, precision/recall
@0.5), mean relative error for comparability with NVIDIA's metric, plus latency and
cost per video-minute.

### 6.4 Order of operations on the box

1. **Serve Edge, validate the deployment** against the paper benchmarks (§6.2)
2. **Capability probe** — does Edge ground events in time, by which mechanism, to
   what precision floor
3. **Measure** VRAM and frames-per-call; set window length from the real number
4. **Sweep** fps and window/stride on a subset
5. **Full matrix** across models and clips
6. Tear down

Steps 1 and 2 gate everything after them. If step 2 fails for Edge, the adapter
switches to Cosmos-Reason2-8B and the matrix is unchanged.

## 7. Steps

| # | Step | Status |
|---|---|---|
| 0 | Verify the model: existence, IDs, licence, VRAM, serving stack, localisation mechanism | **Done** — §1 |
| 1 | Repo layout + HLD design note | **In progress** |
| 2 | Runnable spine: decode -> window -> extract -> merge -> schema -> CLI, with a GPU-free mock path, verified by running it | Not started |
| 3 | **Capability probe on a rented GPU.** Test the brief's claim directly: does Edge localise events in time, by which mechanism, and to what precision? Measure VRAM. Gates everything downstream | Not started |
| 4 | Eval set: synthetic + real clips, hand-labelled, per-video metric isolation | Not started |
| 5 | Measured runs: fps/window sweep, failure analysis against Roboflow's published limits | Not started |
| 6 | README + design note finalised from measured numbers only | Not started |
| 7 | First-run rehearsal on a clean machine, then push | Not started |

---

## 8. Log

- **2026-09-09** — Ran Step 0. Confirmed `nvidia/Cosmos3-Edge` exists (correct repo
  ID has no hyphens), is OpenMDW-1.1 licensed and ungated, and is natively served by
  **stock vLLM** as `Cosmos3EdgeForConditionalGeneration` — vLLM-Omni is only needed
  for the generator tower, which we do not use. Confirmed the timestamp-overlay
  mechanism is real and documented for Cosmos Reason 2 and Cosmos 3, but **not
  stated on the Edge card** — logged as the top risk. Found VANTAGE-Bench, whose
  Temporal Localization task matches this assignment, as a candidate eval source.
  Found Roboflow's own published Cosmos 3 evaluation and recorded its stated
  limitations as targets for our failure analysis.
