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

> **Status: no measurements yet.** Design and verification complete;
> implementation not started. This section is the shape the answer will take — it
> stays empty until real runs fill it, and no number appears here that was not
> measured on the hardware.

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

> Not yet implemented. Planned interface, for reference:

```bash
make preflight   # verify GPU, driver and container toolkit
make probe       # characterise the model: can it ground events, how precisely
make demo        # end-to-end on a sample clip
make eval        # the labelled set → metric table
make sweep       # fps / window / stride frontier
```

Everything runs in Docker — two containers, a CPU `finder` and a GPU `vllm`. The
host needs Docker and the NVIDIA container toolkit, nothing else.

---

## Repository

| File | Contents |
|---|---|
| [`PLAN.md`](PLAN.md) | Requirements, verified facts with sources, experiment plan, open questions |
| [`DESIGN.md`](DESIGN.md) | Architecture, diagrams, mechanisms, validation, the capability probe |
| [`DATASETS.md`](DATASETS.md) | Evaluation sources, licences, failure axes, labelling protocol |
| [`DECISIONS.md`](DECISIONS.md) | Every choice, rejected alternatives, and reversals |
| [`reference/take-home.md`](reference/take-home.md) | The assignment |
