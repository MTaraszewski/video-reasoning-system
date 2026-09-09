# Decisions

Every choice made, why it was made, what was rejected instead, and what would
change it. The brief states that *"we care about your decisions and how you
communicate them at least as much as about the code"* — this is that record.

Entries that were **reversed** are kept, not edited away. A decision that changed
under new evidence is more informative than one that looks like it was right all
along.

Facts cited here are sourced in [`PLAN.md §1`](PLAN.md). Dataset detail is in
[`DATASETS.md`](DATASETS.md); architecture rationale is in [`DESIGN.md`](DESIGN.md).

---

## 1. Models

### Accepted for testing

| Model | Why accepted | What it uniquely answers |
|---|---|---|
| **`nvidia/Cosmos3-Edge`** 4B | The brief recommends it. Open licence (OpenMDW 1.1), **ungated** — no token, no click-through, which matters for the reviewer's first run. Natively served by stock vLLM | Can the *recommended* model actually do the job? The whole assignment turns on this |
| **`nvidia/Cosmos-Reason2-8B`** 8B | Its model card **documents the burned-in-timestamp mechanism**. Edge's does not | Is the *approach* sound, independently of whether Edge supports it? Separates "our design is wrong" from "this model can't do it" |
| **`Qwen/Qwen3-VL-8B-Instruct`** 8B | It is the **base model** Cosmos-Reason2-8B was post-trained from. Same architecture, same size | Does NVIDIA's physical-AI post-training buy anything for temporal localisation, or would the base model do? A controlled comparison, not a leaderboard |
| `nvidia/Cosmos-Reason2-2B` 2B | Same family, 24 GB, cheap | Does capability scale with size within one family? |

The Qwen entry is the only one not from NVIDIA, and it is the one that makes the
others interpretable: without a base-model control, any Cosmos result is a number
without a comparison.

### Rejected

| Model | Why rejected | Would reconsider if |
|---|---|---|
| `nvidia/Cosmos3-Nano` 16B / `Cosmos3-Super` 64B | Far larger than the brief's recommendation and correspondingly more expensive to serve. The assignment is about making the *edge-class* model work, not about finding the best score at any cost | The Edge result is so poor that showing "the family can do it, this size cannot" becomes the finding |
| Hosted commercial APIs | The brief permits them only as *"an explicit fallback path, not the primary engine"*. We have a working open-weights primary, so no fallback is needed | Never, for this deliverable |
| Non-Cosmos video VLMs generally | Breadth is not the question. The brief asks whether *this* model can be productised; a wide model survey would dilute that into a benchmark exercise | — |

---

## 2. Serving

| Decision | Reason | Rejected alternative |
|---|---|---|
| **Stock vLLM** | Cosmos 3 splits into an autoregressive **Reasoner** and a diffusion **Generator**. We need video-in, text-out — the Reasoner only, which stock vLLM serves natively as `Cosmos3EdgeForConditionalGeneration` | **vLLM-Omni** and the `--omni` flag. They exist to serve the *Generator*. We never generate anything, so adopting them would add a fast-moving dependency for capability we do not use |
| **Pinned image tags and vLLM version** | A floating `latest` tag is the most likely cause of "it worked for you, not for us" — the exact failure the brief says it is strict about | Convenience of tracking upstream |
| **Everything in Docker, nothing on the host** | Host requirements reduce to Docker plus the NVIDIA container toolkit. The reviewer's machine state cannot break the run | Local `pip install` — faster to develop, fragile to hand over |
| **GPU gate on `docker run --gpus all`, not host `nvidia-smi`** | One command proves GPU, driver **and** container toolkit together. A host-only check passes while containers still see no GPU — the most common silent failure | Checking `nvidia-smi` on the host |

---

## 3. Datasets

The brief names **no dataset** and asks for *"the events you care about"* on *"any
footage you have the right to use"*. The selection criterion therefore comes from
the following sentence — *"where the limits of their abilities are"* — so clips are
chosen to **stress distinct failure axes**, not to match the brief's three example
phrases.

**The unit of selection is the clip, not the dataset.** Budget is 5–10 clips. Using
six sources costs nothing extra; only clip count drives GPU cost.

### Accepted

| Source | Licence | Why accepted |
|---|---|---|
| **Synthetic** (ours) | ours | The only source with **zero labelling error**. Required to measure the precision floor — a boundary-accuracy number is meaningless against labels that are themselves fuzzy. Also lets us construct any axis a real dataset lacks |
| **MEVA** | **CC BY 4.0** | Ungated, no agreement, fixed camera, redistributable with attribution. 37 activity classes including `person_opens_facility_door`. The strongest combination of rights and fit |
| **VANTAGE-Bench** | eval-only, gated | Warehouse / transportation / smart spaces — the closest domain match, and the only accepted source covering **fast motion and small similar objects**, the two weaknesses Roboflow published. Supplementary tier: we hand-label it, we do not ship it |
| **Roboflow `supervision` assets** | **unstated** | `MILK_BOTTLING_PLANT` covers **absence-of-motion** events, which nothing else does. Crowd scenes too. Fetch-only, never committed |
| **ComplexVAD**, **Street Scene** | **CC BY-SA 4.0** | Fixed-camera, temporally annotated, freely redistributable under share-alike. Cover short rare events |
| **Assembly101** | CC BY-NC 4.0 | The only accepted source with an **overhead camera**, which Roboflow reported as a reliable failure case. Non-commercial licence — a real constraint, stated rather than ignored |

**Kept deliberately: MEVA `G474`, a 352x240 clip among eleven 1080p ones.** The
obvious move was to discard it as an outlier. Instead it becomes its own
**low-resolution axis**: at 352x240 it sits below the 640x360 the model receives, so
its frames are upscaled rather than downscaled. Real deployments have mixed camera
quality, and the brief asks where the limits are — this is free evidence about a
variable nothing else in the set tests. **Condition: reported separately, never
averaged with the 1080p clips**, or resolution gets confounded with every other
difference between cameras.

### Rejected

| Source | Why rejected | Would reconsider if |
|---|---|---|
| **VIRAT** | Distributed under a *"Video Dataset Protection Agreement"*. An agreement the reviewer would have to accept fails the ease-of-use bar | The agreement turned out to permit redistribution |
| **Roboflow Universe** | Image object-detection datasets. No untrimmed video with temporal spans — which is the entire requirement | Video datasets with time annotations appear there |
| **EPIC-KITCHENS** | Egocentric and moving-camera. The system targets fixed cameras; including it would measure the wrong thing | The scope widened to wearables |
| **Pexels / Pixabay / Videvo** | Pexels bars redistribution on other platforms and Videvo bars redistributing unmodified video. Compounding that, stock footage is rarely fixed-camera operational video — a poor domain match regardless of licence | A specific clip were both clearly licensed and domain-appropriate |
| **THUMOS / ActivityNet / Charades-STA** | The standard academic temporal-grounding sets, rejected on domain: YouTube footage, moving cameras, per-video rights unclear at best | The deliverable targeted web video |

---

## 4. Reproduction targets

### Accepted

**Cosmos 3 technical report benchmarks** — arXiv 2606.02800 reports Edge-specific
scores with public ground truth: CVBench 84.9, MMBench-Dev 76.6, RealWorldQA 73.3,
VideoPhy2 40.3.

Accepted because it catches the failure most likely to pass unnoticed: **a serving
bug**. Wrong preprocessing, frame handling or precision yields a model that runs,
answers plausibly, and scores badly — and we would spend the week concluding "Edge
cannot localise events" when the fault was ours. Reproducing ~76.6 on MMBench-Dev
says the deployment is sound *before* any temporal number is trusted.

### Rejected

**Submitting to the VANTAGE-Bench leaderboard.** Rejected because it outsources the
measurement: predictions go to a server, scores return by email on an unknown
turnaround, with a hard budget of 30 submissions per address. The goal is to test
models on our own rented GPU and report numbers we computed. Waiting on someone
else's scoring server is the opposite of that.

**VANTAGE-Bench as a reproduction target** (as distinct from a dataset). Its ground
truth is withheld and scoring is server-side, so its published numbers cannot be
reproduced locally by any means. Its scores — Super 63.01, Nano 60.67, Reason2-8B
54.18 — are context, not a target.

### Demoted, not rejected

**NVIDIA's temporal-localisation recipe** (zero-shot mean relative error 52.0 /
61.3 / 68.45 % on MimicGen). Initially proposed as a headline reproduction, then
demoted — see §6.

---

## 4b. Infrastructure

**Instance: `g7e.2xlarge`** — 1x NVIDIA RTX PRO 6000 Blackwell, 96 GiB VRAM,
8 vCPU, 64 GiB RAM, 1900 GiB local NVMe, **$5.719/hr in `eu-central-1`**
(console-verified; the widely-quoted $3.363 is us-east-1).

| Why | |
|---|---|
| **Blackwell** | Cosmos-Reason2 documents Hopper and Blackwell support. This satisfies it directly, rather than carrying an unsupported-architecture risk |
| **96 GiB VRAM** | Comfortably clears Cosmos-Reason2-8B's documented 32 GB minimum, which is the binding constraint since models are served one at a time |
| **8 vCPU** | Video decoding is CPU work and runs alongside inference. The 4 vCPU alternatives are thin for it |
| **1900 GiB NVMe** | Ephemeral, but ideal for datasets that `make s3-pull` restores in seconds |

**Rejected:**

| Instance | Why not |
|---|---|
| `g6e.xlarge` ~$1.86/hr us-east-1 | Cheaper, and 44.7 GB clears the 32 GB bar — but L40S is **Ada**, which NVIDIA does not list as supported. About $15 saved across the whole project, in exchange for the largest unknown in the runbook |
| `g6.xlarge` ~$0.80/hr us-east-1 | 24 GB runs Cosmos3-Edge fine but not the 8B comparison. Saves money by removing the part that makes a single number interpretable |
| `p5.4xlarge` ~$6.88/hr us-east-1 | Single H100, Hopper, also supported — but 2x the cost with less VRAM than G7e and no advantage at our model sizes |
| `g7.2xlarge` ~$2.52/hr us-east-1 | Blackwell, but RTX PRO 4500 specs unconfirmed and it sits right on the 32 GB line. Not worth guessing to save $0.84/hr |

**Storage: 150-200 GB gp3 EBS root, datasets on the instance NVMe.** Weights and
images must survive stop/start, because re-downloading 50 GB costs more GPU time
than the disk costs in a month. Datasets need not: they restore from S3 in seconds,
and NVMe is faster for repeated decoding.

Full reasoning, download timeline and cost estimates in [`RUNBOOK.md`](RUNBOOK.md).

---

## 5. Scope

| Decision | Reason |
|---|---|
| **CLI is the primary interface**, HTTP optional | Lowest friction for a reviewer running it cold |
| **No control-plane or launcher UI** | Explicitly de-scoped by the brief, and it would add setup friction against the first-try bar. It is dev tooling, not the deliverable |
| **Any UI is an output timeline viewer** | Serves the evaluation and the failure analysis — a way to see where the system fails, not a way to launch it |
| **GPU provisioning stays private dev tooling** | Not what is being graded |
| **Model backend behind a pluggable adapter** | A hedge against one identified risk, not generality for its own sake: the primary model's localisation mechanism is undocumented, so a negative result must cost a config change rather than a redesign |
| **The capability probe is a first-class component** | The brief asks us to *"understand what it actually guarantees"*. Inheriting a capability claim untested would skip the part being graded |
| No pre-start questions to the assessor | Owner's decision. The model choice is justified from primary sources instead |

---

## 6. Reversals

Decisions that changed. Kept deliberately.

### 6.1 VANTAGE-Bench: rejected → accepted

**Originally rejected** on three grounds: ground truth withheld, evaluation-only
licence, and gated access.

**Reversed** because two of the three did not survive examination:

- *Ground truth withheld* — irrelevant. The brief requires us to hand-label anyway,
  so we never needed NVIDIA's labels. The objection was retracted.
- *"Evaluation purposes only"* — arguably **permits** us. Our use is evaluating a
  model, specifically an NVIDIA model, which is what licences of this shape grant.
- *Gated / non-redistributable* — holds, and is why it is a **supplementary tier**
  rather than part of the reproducible core.

The dataset page also confirms download is contemplated: *"When downloaded or used
in accordance with our terms of service..."*

**What this cost:** an initial rejection reached by treating a disqualifying-sounding
fact as decisive without checking it against our actual use. The `model-facts` skill
now carries the rule that caught it — *check licences for the actual intended use*.

### 6.2 NVIDIA temporal-localisation recipe: headline → optional

**Originally proposed** as the main reproduction, on three attractive facts: NVIDIA
publishes **zero-shot** numbers and zero-shot is our setting; the exact prompt is
published; the data is public.

**Demoted** because the benefits turned out to be separable, and most are available
without doing the reproduction at all:

- **The prompt can simply be read and adapted.** Re-running the recipe is not a
  prerequisite for using it. "Learn from the recipe" had been conflated with
  "re-run the recipe".
- **Harness validation is already covered** by synthetic clips with exact
  constructed ground truth.
- The recipe is for **Cosmos-Reason1**, a Qwen-VL model — not our Nemotron-H
  primary — on 7-second clips that never exercise windowing, merging or partial
  events.

What remains is that the control would be *external* rather than self-made. Real,
but not worth days against a one-week deadline.

**Kept:** the published prompt, as the best available answer to the brief's
qualifier *"when prompted correctly"*.

### 6.3 Skills in the repo: recommended out → decided in

Recommended keeping `.claude/skills/` outside the submission to avoid shipping
process detail. **Owner decided to commit them.** Recorded as an owner decision
against the recommendation.

---

## 7. Open — not yet decided

| Question | Blocked on |
|---|---|
| Does Cosmos3-Edge read burned-in timestamps, or localise some other way? | The capability probe, first session on the GPU |
| Which AWS instance | Measured VRAM for the Edge reasoner — no figure is published |
| fps: 4 (documented input rate) or 8 (best in NVIDIA's temporal recipe) | Measurement, not assertion |
| Whether the capability matrix is hand-written or emitted by `make probe` | Preference; emitting it prevents drift |
| A source of **forklift / warehouse** footage beyond VANTAGE | Nothing found yet |
| Licence of the Roboflow `supervision` sample videos | Unstated; worth asking Roboflow directly |
