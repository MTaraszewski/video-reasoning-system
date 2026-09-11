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

## 4c. Overlay size

**`overlay.font_scale = 0.045`** — 16px at the 640x360 the model receives.

Measured, so the trade-off is explicit rather than aesthetic:

| Scale | Font | Frame occluded | JPEG payload |
|---|---|---|---|
| **0.045 (chosen)** | 16px | **1.36%** | 59.5 KB |
| 0.090 | 32px | 4.38% | 59.7 KB |
| 0.135 | 48px | 9.44% | 59.6 KB |

**Token cost does not change with font size** — the frame is the same pixel size
either way — so the only cost of a larger timestamp is occlusion. At 0.135 the box
spans 47% of the frame width and blacks out a fixed region of a fixed camera
permanently.

Chosen by inspection at the model's own resolution: 0.045 is legible, and the
larger sizes buy readability we do not appear to need at the price of a blind spot.

**`UNVERIFIED`: human legibility is not model legibility.** If the probe shows the
model misreading timestamps, raise this before concluding the model cannot read
them — it costs nothing in tokens to try.

---

## 4d. Clip selection — by actor size, not by declared activity

**Screen candidate clips on the actor's median bounding-box height during each
declared event, using MEVA's `.geom.yml`, before downloading any video.**

An event can only be labelled if a human can see it. Nothing else about a clip
matters if that fails, and the annotation that declares an activity says nothing
about how large the actor is in frame.

Calibrated against six clips judged by eye **before this measurement existed**:

| Camera | Median actor h | Hand verdict |
|---|---|---|
| `G326` | **694 px** | two events confirmed |
| `G329` | **295 px** | confirmed (needed zooming) |
| `G331` | **267 px** | suspect, could not verify |
| `G301` | **121 px** | all rejected |
| `G328` | **41 px** | all rejected |
| `G336` | **38 px** | all rejected |

Monotonic across all six. The bands (250 px, 120 px) sit in the gap the data
leaves, rather than at a number chosen for looking round.

**It ranks; it never rejects.** The two errors are not symmetric. A false positive
costs ten seconds looking at a contact sheet. A false negative is *silent* — the
clip never appears in the output and nothing records that it was dropped. `G329`
was nearly lost exactly that way, to an eyeball judgement later overturned by
zooming. Every clip stays in the output; the band annotates.

Cost: `.geom.yml` is 5–70 KB against 56–203 MB per video, so the entire 64-clip
corpus screens for less than the price of downloading one clip.

### Rejected

| Alternative | Why not |
|---|---|
| **Select on "the annotation declares an activity"** (what we did first) | Produced 8 unlabellable candidates from 14, and one clip that lost every event. The declaration is true and useless: it is silent about visibility |
| **Hard-reject below a threshold** | Converts a recoverable mistake into an invisible one. See above |
| **Crop the eval clips to the actor using the annotation's boxes** | Framing the model's input with ground-truth geometry is leakage, and would flatter every number. Acceptable as a labelling aid only — but then we would be labelling events the model still cannot see |
| **Pixel-change screening** (`screen_clips.py`, already built) | Retained for detecting *completely dead* footage, which it does catch. Too blunt here: a person opening a door 40 m away moves a fraction of a percent of pixels, and it cannot tell "active" from "contains your event" |
| **Sample more dates and times** | The intuitive fix, and wrong. `G336` shows `Open_Trunk` at 204 px and `Vehicle_Stopping` at 38 px — the camera is not uniformly bad. Selecting per *event* rather than per *clip* was the actual fix |

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

### 6.4 Clip selection: by declared activity → by actor size

**Was:** choose clips whose `.activities.yml` declares an activity worth labelling.
Reasonable, and it is how the first six clips were picked.

**Now:** rank by the actor's measured size in frame, then choose by event coverage.

**What changed it:** labelling the first set. Eight of fourteen candidates could not
be labelled by anyone — at 26–124 px there is no boundary to read — and
`hospital.G301` lost both its events. The declaration was accurate every time; it
simply does not imply the event is visible.

**What it cost:** one full labelling pass, and it was avoidable. `.geom.yml` sits
beside `.activities.yml` in the same S3 prefix and was being discarded by an `awk`
filter matching only `activities.yml`. The question "how big is the actor?" was
answerable before a single video was downloaded; it was never asked, because the
selection problem had been framed as "which clips contain events" rather than
"which clips contain events we can see well enough to label".

**What it recovered:** `Vehicle_Reversing` at 232 px on `school.G300` — the brief's
*"a forklift reverses"*, which was unlabellable at 41 px on `school.G328`. The
right dataset, the wrong camera, for a reason nothing in the first selection could
have surfaced.

---

### 6.5 GPU: `g7e.2xlarge` chosen -> `g6.2xlarge` measured on

**Was:** `g7e.2xlarge`, RTX PRO 6000 Blackwell, 96 GB, $5.719/hr in eu-central-1.
Chosen to retire a risk: whether vLLM supported Blackwell at all.

**Now:** `g6.2xlarge`, NVIDIA L4, 24 GB, **$1.22249/hr** (verified against the AWS
Pricing API, not estimated). Every measured number in this repository came from
that card.

**What changed it:** `InsufficientInstanceCapacity`, repeatedly, across both
Blackwell and Ada in every availability zone we could reach. The forced move then
retired the original question by accident -- the architecture resolved as
`Cosmos3EdgeForConditionalGeneration` on Ada without incident, so the version pin
was always about vLLM 0.29.0 and never about the GPU generation.

**What it revealed:** the model uses **4.97 GiB of 22.04 GiB**. The 96 GB card was
never needed for a 4B model, and the expensive choice bought nothing this project
used. Recorded because the reasoning that led to it was sound and the conclusion
was still wrong: we sized the instance against an unmeasured risk instead of
against the model.

A side effect worth keeping: capacity hunting on AWS cost more wall-clock than the
measurement did. A quota of 0 in a new region is a support ticket, not a retry.

---

### 6.6 Extraction: one call -> two stages, and confidence from logprobs

**Was:** one call per (window, description), asking the model to find moments
matching the description and to state a confidence in its JSON.

**Now:** stage A asks a neutral yes/no under a prompt stating that most segments
do not contain the action; stage B localises only after a yes; confidence is
P(present) from the yes/no token logprob.

**What changed it:** the measured output. 28 of 81 predictions came back at
confidence exactly 1.0 -- a stated confidence that ranks nothing -- and 7 carried
evidence denying the very event they reported ("Empty hallway with a closed door
and no visible people"). Separately, 13 predictions carried 0.485, which is our
own 0.5 default passed through the merge's noisy-OR: a number we invented,
formatted like a measurement.

**What it did not change, which is the point.** On a 2-clip subsample the two-stage
path produced **the same number of predictions** as the single-stage one. It
removed every degenerate span and gave ten distinct confidence values instead of
three clustered defaults -- but the model said "yes" just as often. So the presence
failure survives the fix, which is what makes it attributable to the model rather
than to our prompt or our harness.

**Rejected:** dropping events whose evidence asserts absence. Tempting -- it would
have removed 7 bad predictions -- but it discards a genuine model behaviour to
make a number look better. The model contradicting itself is a finding; filtering
it out would hide the finding and leave the score unexplained.

---

### 6.8 Confidence for a derived event: three factors, each earned

**Was:** for the states strategy, the fraction of polls inside the interval
agreeing on the target state. A real measurement over data we already held, and
better than a number the model states about itself.

**Now:** `agreement x sharpness x coverage`.

**What changed it:** the same number reached 1.0 on three different kinds of
non-answer, each found by running the thing rather than by reasoning about it.

| what was reported | why it scored 1.0 | the factor added |
|---|---|---|
| a 3-second door as a **42-second event** | one informative poll inside the span, agreeing with itself | — fixed by refusing to interpolate across the gap |
| the same transition bracketed to 1s and to 30s, both at 1.0 | agreement says nothing about how tightly a boundary is pinned | **sharpness** = `step_s` / widest interpolated bracket |
| **"a vehicle door opens", 5.0-97.0s**, four polls holding 92 seconds | all four agreed, and both edges were truncated rather than interpolated, so sharpness was unpenalised too | **coverage** = observed seconds / interval length |
| a single-poll blip at 71.5-72.5s | agreement 1/1, bracket one step wide | also **coverage** -- 0.5s observed of a 1.0s span |

Coverage subsumes a corroboration factor (`min(1, n_polls/2)`) that was considered
and rejected as a second concept doing half the same job. A one-poll event and a
four-poll 92-second event fail for the **same** reason: almost none of the reported
interval was looked at.

**Why this and not the model's own number.** The same argument as 6.6, reached from
the other side. There, a stated confidence was replaced by one derived from token
logprobs. Here there is no yes/no token to take a logprob from, so confidence is
derived from the *structure of the evidence* instead -- how consistent it is, how
tightly it bounds the edges, and how much of the claim was actually observed. All
three are computed from polls already in hand; none costs a model call.

**What it does not change:** intervals. Confidence affects ranking and the greedy
matching order in the eval, not tIoU. A verified good event keeps 0.9; the 92-second
span drops to 0.065.

### 6.7 Model sweep: four planned -> two measured, and why we stopped

**Was:** four models — Cosmos3-Edge, Cosmos-Reason2-2B, Cosmos-Reason2-8B and
Qwen3-VL-8B-Instruct — with the last two as a controlled pair isolating NVIDIA's
physical-AI post-training.

**Now:** two measured, two deliberately not run.

**What changed it.** The 8B models need ~40 GiB reported and the card capacity
gave us reports 22. Qwen3-VL-8B loaded 16.65 GiB of weights, leaving 0.65 GiB for
KV cache against the 2.25 GiB a 16384 context requires.

**The tempting move, and why it is wrong.** They would run at a shorter context
and fewer frames. But Qwen-versus-Reason2 only isolates post-training if both run
under conditions identical to each other AND to the 4B baseline. Constraints we
impose are not a property of the models, and a comparison run under them answers a
question nobody asked. Better an honest gap than a number that looks like a result.

**What the second model did tell us.** Cosmos-Reason2-2B answered 20% of probe
cases at 9.50 s median error, against Cosmos3-Edge's 100% and 3.50 s. Reason2's
model card documents the burned-in timestamp mechanism; Edge's does not mention
timestamps at all. So the most comfortable available explanation — that the brief
recommended a model the technique was never documented for — is ruled out. The
model it IS documented for did worse.

### Rejected

| Alternative | Why not |
|---|---|
| Run the 8B models at reduced frames/context | Breaks the only comparison they exist for. See above |
| Quantise the 8B models to fp8 to fit 24 GB | Would compare a quantised model against two bf16 ones and attribute the difference to post-training |
| Rent a 48 GB card for the pair | Defensible, and the right move if the comparison mattered to the brief. It does not — the brief asks for one model plus its limits, and capacity for 48 GB was not available when we looked |
| Drop the 8B rows from the leaderboard entirely | A dash that says "not run" is information. A missing row is not |

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
