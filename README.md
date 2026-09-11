# Video Reasoning System

**Find events in a video from a plain-language description.** Give it a video and a
phrase — *"a person opens a building door"*, *"a vehicle stops moving"* — and get
back when it happened, with start and end times, in a strict JSON schema.

```
find_events(video, ["a person opens a building door"])
  → [{start_s: 3.5, end_s: 8.5, description: …, confidence: 1.0, partial: false}, …]
```

The client never sees frames, windows, prompts, or the model. That concealment is
the product.

Model: **NVIDIA Cosmos 3 Edge** (4B), served by vLLM v0.29.0 on a single L4.
Everything runs in Docker. Nothing runs on the host.

---

## Quickstart — no GPU, two minutes

```bash
make preflight        # can this host run it at all?
make build            # build the CPU image
make demo             # end-to-end on a generated clip, stub backend
```

`make demo` uses a stub model, so the output is synthetic and the eval harness
refuses to score it. It proves the pipeline, not the model.

## Check everything works before renting a GPU

```bash
make smoke
```

Runs every check that needs no GPU and reports each one: the host, the image, the
pipeline end to end, the labelled data, the decoder, and the **real** model adapter
against a fake endpoint that returns refusals, hallucinations, malformed JSON and
think-blocks. Takes about three minutes.

It does not prove the model works — nothing in it touches real weights.

## Reproduce the measured results — needs one GPU

```bash
make preflight                    # can this host run it at all?
make build                        # build the CPU image
make data-eval                    # rebuild the 8 labelled clips from public sources
make pull                         # pull the pinned vLLM image (large; do it early)
make serve-bg && make serve-wait  # start the model, block until it answers

make eval                                                 # Approach 1   ~1.2 h
make eval STRATEGY=states EVAL_OUT=/out/eval-states.json   # Approach 3   ~4.1 h

make serve-down
```

**That is 5.3 hours end to end.** Add `EVAL_LIMIT=4` to both for a ~3-hour version,
or `EVAL_LIMIT=1` to prove the whole path works in about ten minutes before
committing a GPU to the rest.

`EVAL_OUT=` is not optional on the second command. Both write `/out/eval.json` by
default, so without it the second run silently overwrites the first — a trap that
cost two comparisons during development.

`make data-eval` downloads from MEVA's **public** bucket over plain HTTPS — no AWS
account, no credentials, no `aws` CLI. The clips are not in the repository;
`labels.json` is, and it carries the source file, trim offset and duration for each
clip, which is enough to rebuild the set byte-identically.

`make vars` prints every setting and what it controls. `make help` lists all
targets, each tagged `[any]`, `[local]` or `[gpu]`.

## Re-run any experiment

Each row reproduces one section of [EXPERIMENTS.md](EXPERIMENTS.md). Rows marked
`[gpu]` need `make serve-bg` first.

| # | experiment | command |
|---|---|---|
| 1–2 | capability probe, prompt sweep | `make data-synthetic && make probe` `[gpu]` |
| 2 | overlay legibility | `make frames-sweep` |
| 3 | screen clips by actor size | `make meva-index && make meva-screen` |
| 4 | contact sheets for hand-labelling | `make event-sheets` |
| 5 | **Approach 1, full eval** | `make eval` `[gpu]` |
| 6 | two-stage detect | `make eval DETECT=1` `[gpu]` |
| 7 | second model | `make model-sweep-probe` `[gpu]` |
| 9 | Approach 3, one clip, full length | `make run VIDEO=… QUERIES="…" STRATEGY=states` `[gpu]` |
| 10 | triggered polling | add `TRIGGER=1` to the above `[gpu]` |
| 12 | non-overlapping poll windows | add `SPAN_S=1.0` to the above `[gpu]` |
| 13 | **Approach 3, full eval** | `make eval STRATEGY=states EVAL_OUT=/out/eval-states.json` `[gpu]` |
| 13 | with shared captioning | add `SHARED=1` to the above `[gpu]` |

A concrete single-clip run:

```bash
make run VIDEO=data/eval/2018-03-07.16-50-01.16-55-01.admin.G326.r13.mp4 \
  QUERIES="a person opens a building door" STRATEGY=states
```

Two things to know before comparing runs. `make run` always writes
`out/events.json`, so **copy it between runs** or the second overwrites the first.
And a result depends on which vLLM server instance produced it — comparisons are
only valid inside one `serve-bg` session (experiment 11).

---

## How this started, and how it changed

**The brief** asks for a service that finds events described in plain language. The
interesting problem is not calling a model:

> A video-reasoning model sees a few seconds at a time, and its ability to place
> events in time is asserted but undocumented. How far does that ability go, and
> what has to be built around it?

NVIDIA claims Cosmos 3 Edge *"localises events with timestamps when prompted
correctly"*. Its model card says nothing about temporal localisation. So the claim
was treated as a hypothesis to test.

**Approach 1 — ask the model when the event happened.** Slide a 12-second window,
ask for every moment matching the description, merge across overlaps. This is the
obvious design and it is what was built first.

It scored **mean tIoU 0.0021 and R@1 0.000** across 8 clips and 15 hand-labelled
events. Worse than the score was the shape of the failure: the model reported an
event at nearly the same rate whether one was present (47%) or absent (39%), and 7
of 81 emitted events carried evidence that **denied** the event — *"Empty hallway
with a closed door and no visible people"* — at the model's own stated confidence
of 1.0.

**The diagnosis.** One call was being asked to do three jobs: decide whether the
event is present, locate its boundaries, and format the answer. Probes showed it
can do the third (28 of 30 parsed) and the second when told the event is present
(3.1s median error). It cannot do the first. Mixing all three into one answer meant
a wrong output never said which stage had failed.

**Approach 2 — ask the model to compare two moments.** If it cannot say *when*,
perhaps it can say *which of these two spans* shows the event. A comparison is a
smaller question than a localisation.

It failed for a reason that had nothing to do with vision. Averaged over both
option orders, it scored **exactly 0.00 on every description** — the signature of
answering by *position* rather than content. Asked "A or B" it reliably picked
whichever came last, and reversing the order reversed the answer.

The lesson shaped what came next: **ask the model to describe, not to choose.** A
free-form sentence has no options to be biased by, and the choosing can be done
afterwards in code where it is deterministic.

**Approach 3 — caption, parse, derive.** Ask only what the scene *is*, repeatedly:
*"what state is the door in?"* Parse the answer in code. Derive the event from the
transition between states. The model does perception, which it can do; the code
does temporal reasoning, which it cannot.

On `admin.G326`, the same clip where Approach 1 reported the door opening at
90–102s against a 3.0–5.7s label, Approach 3 returns **3.50–8.50s, tIoU 0.406**.

**Routing.** Not every description decomposes into a binary state. *"Someone hands
an object to another person"* is a relation between two actors; *"a vehicle
reverses"* is motion, which a single window cannot show. Four of the thirteen
labelled descriptions are like that. They are reported as **not expressible** —
never as "no events found", because the failure is a property of the request, not
of the model, and returning a known-bad answer where a client expects a real one is
worse than returning nothing and saying why.

Both approaches stay runnable behind one `find_events`, switched by a flag, so the
comparison is one argument apart rather than one branch apart.

---

## Results

| | Approach 1 (windows) | Approach 3 (states) |
|---|---|---|
| mean tIoU — 8 clips, 15 events | **0.0021** | *not yet scored* |
| mean tIoU — 4 clips, 8 events | **0.000** | *not yet scored* |
| R@1 tIoU≥0.3 | 0.000 | — |
| recall@0.5 | 0.000 | — |
| false-positive rate | 0.80 | — |
| model calls | 1,352 | 1,904 (4 clips, grouped) |
| single clip `admin.G326` | event at 90–102 s | **3.50–8.50 s, tIoU 0.406** |

Approach 1 has **zero overlap with any label** on real footage — not a low score, no
overlap at all. On the same clip where it placed the door opening at 90–102 s
against a 3.0–5.7 s label, Approach 3 returns 3.50–8.50 s.

**But Approach 3 has not been scored across a set.** The harness could not run it
until recently and the run takes hours. One clip is an anecdote; that gap is stated
rather than papered over, and it is the only number missing from this document.

The capability probe, on synthetic clips with exact constructed ground truth:

| prompt | emits a time | median absolute error |
|---|---|---|
| `native` | 80% | **3.11 s** |
| `overlay` | 100% | 3.50 s |
| `terse` | 40% | 7.58 s |

That ~3 s floor is the best boundary accuracy available *before* any windowing, on
clips built to be easy. It bounds everything downstream.

---

## What it cannot do

These follow from the method's primitive — **a persistent binary property of one
object** — and are properties of the design, not defects in it.

| shape | example | works? |
|---|---|---|
| configuration of one object | door open / closed | **yes** |
| posture of one actor | sitting / standing | expressible; fails with several people |
| presence of motion | *"the machine stops moving"* | **expressible** — `moving / stationary` |
| direction of motion | *"a vehicle reverses"* | **no** — a window of positions carries no sign |
| relation between actors | *"hands an object to another person"* | **no** — not a state of any one object |
| compound | *"a person buys something"* | **no** — a sequence, not a state |

Of the brief's three worked examples, *"a person enters through the door"* and
*"the machine stops moving"* are in shape; *"a forklift reverses"* is not.

Also true and worth saying plainly: the state pairs are hand-written; there is no
held-out set, so every clip that produced a number also shaped a threshold or a
prompt; and a result depends on which vLLM server instance produced it (see
[EXPERIMENTS.md](EXPERIMENTS.md)).

---

## What was used, and what was not

Four data sources were selected and assessed. **Two produced numbers**, and the
distinction is worth being explicit about, because an assessed dataset can look
like a used one.

| source | status | used for |
|---|---|---|
| **MEVA** (CC BY 4.0, public bucket) | **used** | the real eval: 8 clips, 15 hand-labelled events |
| **synthetic** (generated here) | **used** | the capability probe — exact constructed ground truth |
| MEVA curated examples | fetched, **not used** | the ceiling test (`eval-control`) was never run |
| `supervision` sample videos | fetched, **not used** | licence unstated, so never a candidate for reported numbers |
| VANTAGE-Bench | assessed, **never fetched** | gated; needs accepted terms and a token |

**Every measured number in this repository comes from MEVA or from synthetic
clips.** Nothing else contributed to a result.

The system is not tied to either. `find_events` takes a video path and a list of
descriptions, so a new dataset needs only clips plus a `labels.json` in the same
shape — see [DATASETS.md](DATASETS.md#reproducing-the-eval-set). What does *not*
transfer automatically is `states.json`: the description → state-pair mapping is
hand-written, so a new domain needs that written too. Deriving it from the
description is one cheap text call and is not built.

## Not done, and why

Traceability for everything planned or implied that does not exist. An unrun
experiment quoted as a result is the worst kind of error, so these are named.

| not done | why | consequence |
|---|---|---|
| **Approach 3 scored across a set** | the harness could not run it until recently; the run takes hours | the one hole in the results table |
| ceiling test — activity name burned into the frame | not built out | cannot separate "cannot recognise events" from "cannot see at this resolution" |
| Qwen3-VL-8B vs Cosmos-Reason2-8B | neither fits in the L4's 22 GiB with a 48-frame window | the comparison isolating NVIDIA's post-training stays open |
| held-out set | needs more labelled clips, not a post-hoc split | every clip that produced a number also shaped a prompt or threshold |
| deriving state pairs from the description | one cheap text call; unbuilt | a human is still in the loop, once per new description |
| two-stage trigger — locate brackets, then sweep inside them | designed from the triggered-polling result, not built | triggered polling is 10× cheaper and loses the event, so it stays off |
| instant-vs-span metric in `metrics.py` | interval tIoU already works | transition scoring lives in a script rather than the harness |
| captioning in the stub and replay backends | the recorder hooks `extract()` only | Approach 3 cannot be exercised without a GPU |
| `make sweep`, `make viz` | marked planned in the Makefile | fps/window frontier and timeline rendering unavailable |
| tests, CI, linters, scaling, security | the brief states these are not evaluated | deliberate — see [DESIGN.md §15](DESIGN.md) |

## Where the detail lives

Everything above is the summary. The evidence is in:

| document | what it holds |
|---|---|
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | every experiment run, why it was run, what it showed |
| [DESIGN.md](DESIGN.md) | mechanisms, schema, the long argument for each choice |
| [DECISIONS.md](DECISIONS.md) | choices made, alternatives rejected, decisions that reversed |
| [DATASETS.md](DATASETS.md) | data sources, licences, how the eval set was chosen |
| [RUNBOOK.md](RUNBOOK.md) | operating the GPU box, costs, failure recovery |
| [PLAN.md](PLAN.md) | dated worklog |

## Repository

```
src/video_reasoning/    the service
  core.py               find_events — both strategies
  states.py             caption → parse → derive (Approach 3)
  motion.py             change signal, for triggered polling
  evaluate.py           the scoring harness
  metrics.py            temporal IoU, recall, precision, false-positive rate
  backends/             model adapter: vllm, stub, replay, fake
scripts/                data preparation, probes, one-off experiments
make/                   every operation, tagged [any] / [local] / [gpu]
data/eval/labels.json   the hand-labelled set — the only data in git
states.json             description → state pair (hand-written)
config.yaml             every tunable, with its reason
```
