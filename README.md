# Video Reasoning System

**Find events in a video from a plain-language description.** Give it a video and
a phrase — *"a person enters through the door"*, *"the machine stops moving"* — and
get back when it happened, with start and end times, in a strict JSON schema.

```
find_events(video, ["a forklift reverses"]) → [{start_s, end_s, description, confidence}, …]
```

The client never sees frames, windows, prompts, or the model. That concealment is
the product.

---

## The research question

The interesting problem is not calling a model. It is this:

> **A video-reasoning model can only look at a few seconds at a time, and its
> ability to place events in time is asserted but undocumented. How far does that
> ability actually go, and what has to be built around it to make it usable?**

NVIDIA's **Cosmos 3 Edge** is claimed to *"localise events with timestamps when
prompted correctly"*. Its model card says nothing about temporal localisation. So
the claim is treated here as **a hypothesis to test**, not a specification to build
on.

---

## How this was approached

As an investigation, not a build. Five stages, in order:

**1 — Establish what is actually true.** Every model, licence, VRAM figure and
serving command verified against a primary source before it entered a document or a
decision. Claims are tagged: `?` unknown · `D` vendor-documented, untested here ·
`B` asserted by a third party · `PASS`/`FAIL` measured here. Nothing is promoted
without evidence. → [`PLAN.md §1`](PLAN.md)

**2 — Design around the uncertainty.** The model sits behind an adapter precisely
because its localisation mechanism is unconfirmed, so a negative result costs a
config change rather than a redesign. → [`DESIGN.md`](DESIGN.md)

**3 — Measure the model before measuring the system.** A capability probe runs
first, on synthetic clips with *exact* constructed ground truth, to find the
**precision floor** — the best boundary accuracy achievable before any windowing is
layered on. Without it, a mediocre score is unattributable: model error and
pipeline error cannot be told apart. → [`DESIGN.md §10`](DESIGN.md)

**4 — Choose evidence that can disprove things.** Clips are selected to stress
*distinct failure axes* — fast motion, small similar objects, absence of motion,
overhead views, short events, crowds — not to flatter the system. Three of those
axes come from weaknesses Roboflow itself published about Cosmos 3.
→ [`DATASETS.md`](DATASETS.md)

**5 — Report what was found, including the negative results.** Every choice, the
alternatives rejected, and the decisions that reversed under new evidence.
→ [`DECISIONS.md`](DECISIONS.md)

---

## Results

> **Status: measured.** `nvidia/Cosmos3-Edge` was served on an NVIDIA L4 via vLLM
> 0.29.0 and run against the hand-labelled set — 1,352 model calls, plus a
> capability probe on synthetic clips and a second probe on the real ones. Every
> number below was measured on that hardware. Nothing here is projected.
>
> **The result is negative, and that is the result.** The brief asks where the
> limits of these models are. We found a specific, reproducible one.

### Leaderboard — models × temporal grounding

| Model | Size | R@1 @0.3 | R@1 @0.5 | R@1 @0.7 | mean tIoU | mean rel. err | s / video-min |
|---|---|---|---|---|---|---|---|
| `nvidia/Cosmos3-Edge` | 4B | **0.000** | **0.000** | **0.000** | **0.002** | **3.151** | **268** |
| `nvidia/Cosmos-Reason2-8B` | 8B | — | — | — | — | — | — |
| `nvidia/Cosmos-Reason2-2B` | 2B | — | — | — | — | — | — |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | — | — | — | — | — | — |

Only the first row ran. The remaining three are **not measured**, and the
Qwen-versus-Reason2 comparison — which would isolate what NVIDIA's physical-AI
post-training buys for temporal localisation — remains an open question, not a
finding. NVIDIA's own target for mean relative error is <0.30; we measured 3.151.

### The finding: it cannot tell whether an event is present

Every description was asked of every clip — 104 (clip, query) pairs, of which 15
have a real answer. Asking only each clip's own queries would have measured
nothing about false positives.

| | pairs | model reported an event |
|---|---|---|
| event **is** present | 15 | 7 (**47%**) |
| event is **absent** | 89 | 35 (**39%**) |

**It reports an event at close to the same rate whether or not one is there.**
That, not boundary imprecision, is why every tIoU-based number is near zero. With
n this small the difference is not statistically strong — but the direction is
clear, and it is corroborated by the probes below.

Concretely: asked for *"a person gets out of a vehicle"* in an indoor stairwell,
it answered **99–111 s with confidence 0.87**, explaining that the man on the
stairs *"suggests he is exiting the vehicle."*

### The probes: it localises when told, but cannot detect

A capability probe runs **one window over the whole clip** — no windowing, no
merging — so whatever error remains belongs to the model.

| | synthetic clips | real footage |
|---|---|---|
| answered at all | **100%** | **13%** |
| median boundary error | **3.50 s** | 9.01 s |

On synthetic clips, where ground truth is exact by construction and the event is
guaranteed present, the model **does** localise — 3.5 s median error, answering
every time. On real footage, with the event equally guaranteed and centred in the
frames shown, it declined on **13 of 15 cases**.

So the capability is real but conditional: *given that an event is there, it can
say roughly when.* It cannot establish the "given".

**The 3.5 s figure does not generalise, and we withdraw it as a model
characteristic.** It describes the model's behaviour on synthetic stimuli. The
distinction only became visible by running the same probe on both.

### Where it breaks, by failure axis

Measured on synthetic clips whose ground truth is exact:

| Axis | Median boundary error |
|---|---|
| visually similar distractor | 2.00 s |
| long event | 3.00 s |
| baseline | 4.25 s |
| short event | 5.40 s |
| **absence of motion** | **7.74 s** |

*"The machine stops moving"* — one of the brief's own three examples — is the
worst axis by a factor of two. Short events score 5.4 s error on events lasting
under a second.

### This confirms the vendor's own published evaluation

Roboflow published their [Cosmos 3 evaluation](https://blog.roboflow.com/cosmos-3-vision/):
strong on **slow-changing states** in fixed-camera footage, weak on **fast motion
and small objects**, and — the operative finding — *"splitting the region of
interest per gate and running inference on each gate separately beat one combined
call"*, with the principle *"isolate each zone and point the model at the state
that changes slowly, not the motion that changes fast."*

Our configuration was whole-frame, no region of interest, motion-event queries,
actors 26–787 px in wide surveillance shots. That is their worst-case
configuration on every axis they name. **Our near-zero result is an independent
reproduction of the caveats the model's own evaluators published**, on different
footage, with a different harness.

The honest reading is therefore not "this model does not work". It is: *asked in
the way a client would naturally ask — open-vocabulary event descriptions over a
whole frame — it does not work, and the configuration that does work is a
different question shape than the one the brief poses.*

### What we got wrong along the way

Recorded because the debugging is part of the answer:

- **Our harness invented confidence.** 13 of 81 predictions carried 0.485 — our
  own 0.5 default laundered through the merge's noisy-OR. An invented number
  wearing the shape of a measurement. Fixed: confidence now comes from a yes/no
  logprob, or is absent.
- **Degenerate spans counted as events.** 13 zero-length points and 33
  whole-window spans — 57% of all predictions were non-answers in an interval's
  clothing. Now rejected and counted.
- **A hypothesis we withdrew.** Twelve samples suggested the model always reports
  the tail of its window. All 81 refuted it (mean position 40%, spread evenly).
  The twelve came from a prompt we had written that afternoon.

### The evaluation set — built, hand-labelled

**8 clips, 120 s each, 15 events labelled by hand**, from MEVA (CC BY 4.0), in
`data/eval/labels.json`. All three of the brief's worked examples are covered:
*"a person enters through the door"*, *"a forklift reverses"* (`Vehicle_Reversing`)
and *"the machine stops moving"* (`Vehicle_Stopping`). Event durations run 1.4 s to
13.6 s, median 2.3 s, so short and long events are both tested on real footage.

**First measured finding, and it is about the data rather than the model.** An
event can only be labelled if a human can see it. MEVA's `.geom.yml` gives
per-frame actor boxes, and median actor height ranks **monotonically with six
verdicts reached by eye before that measurement existed**:

| Median actor height | Could a person label it? |
|---|---|
| 694 / 295 / 267 px | yes |
| 121 / 41 / 38 px | no — nothing to read a boundary from |

Three clips are therefore kept **unlabelled and excluded from every score**: at
26–124 px there is no honest ground truth, so a number computed against them would
measure the labeller, not the model. They are not wasted — MEVA declares events in
them that no person can verify, which makes them a test for whether the model
invents confident localisations when the evidence is absent. Reported separately,
never pooled.

**The clips are not in this repository; the labels and their provenance are.**
400 MB of someone else's dataset does not belong in a git history, and shipping it
would ask you to trust our copy. Instead `labels.json` carries, per clip, the source
file, the trim offset and the duration — and `make data-eval` fetches the public
sources and re-cuts them. The rebuild was verified **bit-identical**: the same
SHA-256 over 60 sampled frames as the clips the labels were made against. So every
number reported here is reproducible from a published dataset, on your machine,
without taking our word for the footage.

The screen that predicts labellability from annotation files alone — before any
video is downloaded — is `scripts/screen_geom.py`. It **ranks and never rejects**: a false
positive costs ten seconds looking at a contact sheet, a false negative is silent.
→ [`DATASETS.md`](DATASETS.md)

### Leaderboard — models × temporal grounding

| Model | Size | R@1 @0.3 | R@1 @0.5 | R@1 @0.7 | mean tIoU | mean rel. err | s / video-min | $ / video-min |
|---|---|---|---|---|---|---|---|---|
| `nvidia/Cosmos3-Edge` | 4B | — | — | — | — | — | — | — |
| `nvidia/Cosmos-Reason2-8B` | 8B | — | — | — | — | — | — | — |
| `nvidia/Cosmos-Reason2-2B` | 2B | — | — | — | — | — | — | — |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | — | — | — | — | — | — | — |

The last row is the control: it is the base model Cosmos-Reason2-8B was
post-trained from, so the difference between those two rows isolates what NVIDIA's
physical-AI training actually buys for this task.

### Figures the runs will produce

| Figure | What it answers |
|---|---|
| **Precision floor** — boundary error vs event duration | How accurately can the model place a boundary at all? |
| **Accuracy/cost frontier** — tIoU vs sampling fps | Is 8 fps worth double the tokens over 4? |
| **Window/stride sweep** — recall vs overlap | How much overlap does boundary-straddling recovery actually need? |
| **Per-axis breakdown** — score by failure axis | *Where does it break?* The question the brief actually asks |
| **Model × axis heatmap** | Does a 4B edge model fail differently from an 8B, or just more? |

---

## Running it

Everything runs in Docker. The host needs Docker, and the NVIDIA container toolkit
only for the model server — the rest is CPU work.

**Works today, no GPU required:**

```bash
make preflight    # can this machine run anything?
make build        # build the finder image
make demo         # end-to-end -> events JSON, using the stub backend
                  #   generates its own clip if data/ is empty, so this is
                  #   the whole first-run sequence: clone, preflight, build, demo
make data         # generate the full synthetic set with exact ground truth
make verify-data  # do the clips actually show what their labels claim?
make probe        # characterise a model: can it ground events, how precisely
make plan  VIDEO=clip.mp4 QUERIES="a door opens"   # what a run would cost
make run   VIDEO=clip.mp4 QUERIES="a door opens;a vehicle stops"
make frames VIDEO=clip.mp4        # see the timestamped frames the model receives
make fake-test                    # the real vLLM backend against a fake endpoint
```

`make fake-test` is worth knowing about: it exercises the actual model-facing code
— HTTP, reasoning-block parsing, time reconciliation, the model-identity check —
against an endpoint that returns reasoning blocks, truncated responses, refusals
and hallucinated timestamps. No GPU involved.

**Needs a GPU** — or any real endpoint. This is the full sequence on a bare box:

```bash
make preflight-gpu       # is a GPU visible to CONTAINERS, not just the host?
make build && make pull  # finder image, then the pinned vLLM image (multi-GB)
make data-eval           # rebuild the 8 labelled clips from their public sources
make serve-bg            # start the model, block until it answers
make probe               # the gating question: can it ground events, how precisely
make eval                # the labelled set -> metric table
make eval-control        # ceiling test: can it find events labelled on-screen?
make down                # stop the containers before you stop paying
```

`make data-eval` needs no AWS account, credentials or CLI — it fetches over HTTPS
from MEVA's public bucket and re-cuts each clip to the exact offset and duration
`labels.json` records.

Build and pull come before everything because `probe`, `eval` and `data-eval` all
run inside the finder container. A metered box should not idle on a build it could
have done first.

`probe` and `eval` are built and exercised end to end against a fake endpoint;
they need a real model to produce a real answer, not to run.

`make help` lists every target, tagged `[local]` / `[gpu]` / `[any]`, with the
end-to-end workflow.

---

## Repository

| File | Contents |
|---|---|
| [`PLAN.md`](PLAN.md) | Requirements, verified facts with sources, experiment plan, open questions |
| [`DESIGN.md`](DESIGN.md) | Architecture, diagrams, mechanisms, validation, the capability probe |
| [`DATASETS.md`](DATASETS.md) | Evaluation sources, licences, failure axes, labelling protocol |
| [`DECISIONS.md`](DECISIONS.md) | Every choice, rejected alternatives, and reversals |
| [`RUNBOOK.md`](RUNBOOK.md) | Provisioning a GPU: instance sizing, what downloads when, cost, risks |
| the assignment brief | Not included — it is the client's document, not ours to republish. Requirements are traced in `PLAN.md` |
