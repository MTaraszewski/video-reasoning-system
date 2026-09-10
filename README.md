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

The first six columns come from the **eval** — real clips, hand labels, full
cross-product. The last two come from the **probe** — synthetic clips, exact
ground truth, event guaranteed present. They are different experiments and are
not comparable with each other; both are shown because a model can be measured by
one and not the other.

| Model | Size | R@1 @0.3 | R@1 @0.5 | R@1 @0.7 | mean tIoU | mean rel. err | s / video-min | probe: answered | probe: median err |
|---|---|---|---|---|---|---|---|---|---|
| `nvidia/Cosmos3-Edge` | 4B | **0.000** | **0.000** | **0.000** | **0.002** | **3.151** | **268** | **100%** | **3.50 s** |
| `nvidia/Cosmos-Reason2-2B` | 2B | not run | not run | not run | not run | not run | not run | **20%** | **9.50 s** |
| `nvidia/Cosmos-Reason2-8B` | 8B | not run | not run | not run | not run | not run | not run | not run | not run |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | not run | not run | not run | not run | not run | not run | not run | not run |

Both probe columns are the `overlay` prompt at 4 fps, so those two rows are
like-for-like. "not run" is literal — no cell here is estimated.

Cosmos3-Edge is the only model given the full eval. **Cosmos-Reason2-2B was
probed** (result below); the two 8B models were not run at all — they need ~40 GiB
and the card we obtained reports 22. So the Qwen-versus-Reason2 comparison, which
would isolate what NVIDIA's physical-AI post-training buys for temporal
localisation, remains an open question rather than a finding. NVIDIA's own target
for mean relative error is <0.30; we measured 3.151.

### A second model does not rescue it

The obvious hypothesis after the first result was that we had picked the wrong
model. `Cosmos3-Edge`'s card does not mention timestamps at all; the brief
asserted the capability. `Cosmos-Reason2`'s card **does** document the burned-in
timestamp mechanism. If the approach worked there, the finding would have been
"the brief recommended a model this technique isn't documented for".

It does not. On the same synthetic clips, same prompt, same sampling rate:

| Model | Architecture | answered | median boundary error |
|---|---|---|---|
| `nvidia/Cosmos3-Edge` (4B) | Nemotron-H | **100%** | **3.50 s** |
| `nvidia/Cosmos-Reason2-2B` (2B) | Qwen3-VL | **20%** | 9.50 s |

The model whose card documents the mechanism did **worse** — declining 4 of 5
cases where the event was present by construction. vLLM resolves Reason2-2B as
`Qwen3VLForConditionalGeneration`, confirming the Reason family is Qwen3-VL with
NVIDIA post-training on top, so this is a second *architecture* failing too.

Three variables differ at once — size (4B vs 2B), architecture, and whether the
mechanism is documented — so this is a data point, not a controlled comparison.
What it does rule out is the comfortable explanation: the failure is not a quirk
of the one model the brief named.

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

### Independently corroborated

Roboflow published their own [Cosmos 3 evaluation](https://blog.roboflow.com/cosmos-3-vision/)
and report the same weaknesses we measured — **fast motion and small objects**,
with the model strongest on slow-changing states. Different footage, different
harness, same conclusion. That is worth more than another run of our own.

They also report a remedy: isolating a region of interest and running inference
per region. **We implemented and measured it, and it did not help here** — a fixed
crop applied at native resolution took `admin.G329`'s score range from 0.12 down
to 0.07. The hypothesis behind it, that subject size in frame is the limit, was
then disproved directly: a car door succeeds at 322 px where a person in a doorway
fails at 295 px.

### A second approach: poll the state, derive the event

The first approach failed specifically — the model can *time* an event it is told
is present and cannot establish that it is present. So the second stops asking it
to do either. Ask what the scene **is**, repeatedly, in both option orders to
cancel position bias, take confidence from the token logprobs, and call the event
where the score departs from the clip's own baseline. **The model is never asked
what time it is**; the timestamp comes from our sampling grid.

| clip | question | outcome |
|---|---|---|
| `admin.G326` 03-07 | building door closed / open | **tIoU 0.46** |
| `admin.G326` 03-12 | building door closed / open | **tIoU 0.26** |
| `school.G300` | car door closed / open | **tIoU 0.54** / **0.55** — one interval, two labels |
| `school.G300` 03-13 | door question, no door in scene | **no detection** — true negative |
| `admin.G329`, `school.G423`, `bus.G340`, `school.G300` motion | four questions | misses |

Three distinct detections, mean tIoU **0.42** where it fires, against **0.002**
for the first approach across its entire eval.

**We proposed two scope rules and falsified both.** *"Binary configurations work,
presence and motion fail"* died on a person sitting down. *"Doors work"* died on a
car door that didn't. Actor size, object size and object class were each
contradicted by a later test. Six questions is not enough to establish a rule, and
we are not going to invent one from three successes.

### The finding that reframes all of it

Every probe above capped generation at 1–4 tokens and read a logprob — using a
reasoning model as a one-token classifier. Letting it describe the scene first, then
scoring the description, gave this:

> t=2 — *"The door is **closed** in all frames"*
> t=4 — *"Sixth frame: **a person is opening the door**"*
> t=6 — *"The door is **open** in some frames, showing a person inside"*
> t=8 — *"the door seems to be **closed**"*

Against a hand label of 3.0–5.7 s, correct at every timestep. **The perception was
there the whole time, and four framings were discarding it.**

Two defects sat between it and the score, both ours: we asked *"describe the door"*
and got appearance rather than state, so the classifier scored **+0.91 for "open"
on text saying "closed in most frames"**; and the server concatenates the model's
reasoning with its answer, so the classifier was reading it think aloud. Both are
fixed. **Whether that recovers the signal end to end is not yet measured**, and is
not claimed.

That also explains the `G423` miss without a new theory: the description says *"a
person standing near a table in the hallway"* — in a scene with several people. The
subject was ambiguous, so the question was never well posed.

**Status:** these are probe scripts, not pipeline code. `find_events` still runs
the first approach. Nine questions across five clips is a characterisation, not an
evaluation. → [`DESIGN.md §14a`](DESIGN.md)

### A third approach: caption, parse, derive

Following that finding to its conclusion gives a design with **no forced choice, no
logprobs and no threshold** — the three things every failure above traced back to.

Per timestep, ask the model what state the subject is in, free-form. Read the state
out of its own words deterministically. The event is the transition between
consecutive states.

On `admin.G326`, label 3.0–5.7 s, 1-second steps:

```
t=0–4    closed
t=5      closed   ← sees the change, reverses its direction
t=6–8    open
t=9      closed
t=11–17  closed
```

Last `closed` at t=5, first `open` at t=6 — the **transition sits at t≈5.5 s,
inside the label**, and the door returns to closed at t=9, which matches the
footage independently. Agreement of roughly **0.3 s**, from a model whose best
synthetic boundary error was 3.5 s and which under the first approach could not
answer on real footage at all.

The model does the one thing it has done well throughout: describe what it sees.
Everything after that is code — and the parse is string matching rather than a
model call, because an order-averaged text classifier scored **exactly 0.00 on
every description**, the signature of choosing purely by position.

**One failure mode, recorded because it is instructive:** at t=5 it says *"The door
starts in an open state and closes"* — right moment, wrong direction. Seeing a
change and getting its sign backwards is a more tractable problem than not seeing
it.

**How this must be scored.** Interval tIoU is the wrong measure and understates it:
a state timeline answers *"when was it open"* (6–9 s) while the labels answer *"when
did it open"* (3.0–5.7 s). The right measure is the **transition instant against the
label's span**.

### Scored across the labelled set

`make transitions` runs this over every labelled event and scores the **transition
instant against the label's span**:

| | event | label | transition |
|---|---|---|---|
| HIT | `G329` enters through door | 3.0–4.8 | 3.5 s |
| HIT | `G326` opens building door | 3.0–5.7 | 5.5 s |
| HIT | `G326` enters through door | 5.2–7.5 | 5.5 s |
| HIT | `G340` gets into a vehicle | 3.0–6.8 | 5.5 s |
| HIT | `G300` vehicle door opens | 9.0–12.7 | 10.5 s |
| HIT | `G326` comes out through door | 3.0–5.3 | 3.5 s |
| miss | `G300` vehicle stops moving | 9.1–10.7 | no transition |
| miss | `G300` gets out of a vehicle | 11.5–13.7 | 0.3 s outside |
| miss | `G423` sits down | 3.0–5.0 | 2.0 s away |
| miss | `G423` stands up | 37.6–39.0 | no transition |

**6 of 10**, and two of those hits — `G329`, `G340` — are clips the previous
approach could not touch at all.

**Three caveats that belong next to that number, not below it:**

**Five of the fifteen labelled events aren't scoreable.** No binary state pair
expresses *"a vehicle reverses"*, *"a person buys something"*, *"someone hands an
object to another person"* or *"a vehicle drops someone off"*. The first is the
brief's own forklift analogue. A denominator of 10 is not full coverage.

**There is no held-out set.** `G326`, `G329`, `G423`, `G300` and `G340` were each
used to develop a prompt, a threshold or a state pair before being scored. The
state mappings in `states.json` are hand-written by someone who had already seen
which framings worked. 6/10 is measured on the data the method was tuned against,
and no post-hoc split fixes that — only more labelled clips, held back and scored
once.

**The system still needs a human in the loop.** A client types a sentence; someone
has to translate it into a state pair before anything runs. Deriving that
automatically is one cheap text call and is not built.

→ [`DESIGN.md §14a`](DESIGN.md)

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

### Figures

| Figure | What it answers |
|---|---|
| **Precision floor** — boundary error vs event duration | How accurately can the model place a boundary at all? |
| **Accuracy/cost frontier** — tIoU vs sampling fps | Is 8 fps worth double the tokens over 4? |
| **Window/stride sweep** — recall vs overlap | How much overlap does boundary-straddling recovery actually need? |
| **Per-axis breakdown** — score by failure axis | *Where does it break?* The question the brief actually asks |
| **Model × axis heatmap** | Does a 4B edge model fail differently from an 8B, or just more? **Not produced** — the 8B models need a 48 GB card |

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
