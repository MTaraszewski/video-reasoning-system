# Experiments

Every experiment run, why it was run, and what it showed — including the ones that
failed and the ones that overturned an earlier conclusion. Ordered as they
happened, because several only make sense as answers to the previous one.

Each entry: **the question**, the method, the result, and what changed because of
it. Where something was *not* run, it says so.

---

## 1 — Can the model place an event in time at all?

**Why.** NVIDIA claims Cosmos 3 Edge *"localises events with timestamps when
prompted correctly"*; its model card never mentions temporal localisation. Building
a windowing system on an unverified claim risks discovering at the end that the
foundation was never there.

**Method.** Synthetic clips with *exact* constructed ground truth — a shape appears
at a known millisecond — so model error is not confounded with label error.
`make probe`.

**Result.** It emits a time, and the time is roughly right.

| prompt | emits a time | median absolute error |
|---|---|---|
| `native` | 80% | **3.11 s** |
| `overlay` | 100% | 3.50 s |
| `terse` | 40% | 7.58 s |

**What changed.** The claim survives, weakly: ~3 s is the precision *floor* on
clips built to be easy. Every later result is bounded by it. `terse` was dropped.

---

## 2 — Can the model read a timestamp burned into the frame?

**Why.** The overlay strategy only works if the model can read what is drawn. Human
legibility is not model legibility, and assuming it would have made every timestamp
error look like a reasoning failure.

**Method.** One frame rendered at five font scales (`make frames-sweep`), then the
`overlay` prompt measured against `native`.

**Result.** `overlay` reaches 100% emit against `native`'s 80%, but a *worse* median
error (3.50 s vs 3.11 s). Reading the number makes the model more willing to answer
and no more accurate.

**What changed.** `native` became the default. The overlay stays available and its
occlusion cost (1.36% of frame at scale 0.045) is documented rather than guessed.

---

## 3 — Which real clips are worth hand-labelling?

**Why.** Labelling is the expensive, irreversible part. MEVA is 5-minute clips of
mostly-empty car parks; picking badly means measuring the model on footage where
nothing is visible at 640×360.

**Method.** Fetch MEVA **annotations only** (no video), then rank candidate clips by
the on-screen pixel height of the actors during their declared activity, taken from
the published `.geom.yml` bounding boxes. `make meva-index`, `make meva-screen`.

**Result.** Actor height ranged 26–787 px. The ranking agreed with six hand verdicts.

**What changed.** The eval set became 8 clips chosen by a measurable property rather
than by eye. It also made the small-actor limitation quantitative instead of
anecdotal.

---

## 4 — Do the labels actually say what the footage shows?

**Why.** MEVA's annotations are the ground truth, and a wrong label makes a correct
prediction score as an error.

**Method.** A dense contact sheet per candidate event (`make event-sheets`),
reviewed by hand, with disagreements resolved by zooming into the source video.

**Result.** Two of my own verdicts were wrong and were overturned by zooming —
`G326` ("the door is shut until 5.0s, so the label starts too early") tracked the
door *panel* rather than the act of opening; a person is visible behind the glass
working the handle from 3.0s, exactly as labelled. Contact sheets are biased
**late** on event starts.

**What changed.** 8 clips, 15 events, each marked `confirmed_by_hand` with a note
saying how it was decided. Sheets stopped being trusted alone.

---

## 5 — Approach 1: ask the model when the event happened

**Why.** The obvious design, and the one the brief's phrasing suggests.

**Method.** 12-second windows, 9-second stride, 4 fps; one call per (window,
description) asking for every moment matching it; merge across overlaps.
`make eval`.

**Result.** **mean tIoU 0.0021, R@1 0.000, false-positive rate 0.80**, 81
predictions against 15 truths, 1,352 calls.

Worse than the score was the shape of the failure. The model reported an event at
nearly the same rate whether one was present (47%) or absent (39%). Seven of 81
emitted events carried evidence that **denied** the event — *"Empty hallway with a
closed door and no visible people"* — at a stated confidence of 1.0. Confidence
clustered on three values (0.97 ×28, 0.485 ×13, 0.96 ×12), so it ranked nothing.

**What changed.** This is the result that redirected the project. A single call was
being asked to decide presence, locate boundaries, and format JSON. Probes showed
it can format (28 of 30 parsed) and can localise when told the event is present
(3.1 s median). It cannot decide presence. Mixed into one answer, a wrong output
never said which stage had failed.

---

## 6 — Is it the model, or the harness?

**Why.** "Reports events at the same rate whether present or absent" could equally
describe a model that cannot detect, or a pipeline that manufactures events out of
non-answers. Those need different fixes.

**Method.** Split detection from localisation: a constrained yes/no first, with
confidence taken from the **token logprobs** rather than a number the model states;
localise only on yes. Reject evidence that asserts absence, zero-length points, and
spans covering ≥95% of the window.

**Result.** The harness was manufacturing positives — 10 of 81 predictions had
evidence asserting absence and were emitted anyway; 16% were zero-length, 40% were
whole-window.

**What changed.** The numbers became *interpretable*, which is different from
better. A positive now requires the model to say yes under a prompt where no is
easy, and to return a non-degenerate interval.

---

## 7 — Is it this model, or this class of model?

**Why.** The comfortable explanation for a near-zero score is "wrong checkpoint".

**Method.** A model registry and serial sweep (`make model-sweep`), scoring each on
the same synthetic probe.

**Result.** **Cosmos-Reason2-2B: 20% emit, 9.50 s median error**, against
Cosmos3-Edge's 100% / 3.50 s at identical prompt and fps. vLLM resolves Reason2-2B
as `Qwen3VLForConditionalGeneration`, so that is a second *architecture* failing.
Notably, Reason2's card documents the burned-in-timestamp mechanism and Edge's does
not — the model the technique is documented for did worse.

**Not measured.** Both 8B models. Qwen3-VL-8B loaded 16.65 GiB of weights on the
L4's 22.04 GiB usable, leaving 0.65 GiB for KV cache against the 2.25 GiB a 16k
context needs — vLLM capped it at 4,720 tokens, too small for a 48-frame window.
Running them at fewer frames would have broken the comparison they exist for. **The
Qwen-8B-versus-Reason2-8B comparison, which would isolate NVIDIA's post-training,
remains open.**

Three variables move at once here (4B vs 2B, two architectures, documented vs not),
so it is a data point, not a controlled comparison. What it rules out is the
comfortable explanation.

---

## 8 — Approach 2: ask the model to compare two moments

**Why.** If the model cannot say *when* something happened, perhaps it can say
*which of two moments* shows the event — a comparison is a much smaller question
than a localisation, and a binary answer is easy to score.

**Method.** Show two short spans and ask which one contains the described event,
or whether the state differs between them. Probe script, never pipeline code.

**Result.** It failed for a reason that had nothing to do with vision. Averaged
over both option orders, the model scored **exactly 0.00 on every description** —
the signature of answering by *position* rather than by content. Asked "A or B" it
reliably picked the last option offered; reversing the order reversed the answer.

**What changed.** Two things, and the second matters more than the first.

The immediate lesson was to stop asking the model multiple-choice questions.
Option-order bias cannot be prompted away, and averaging two orders costs double
for an answer that carries no information.

The deeper lesson shaped Approach 3: **ask the model to describe, not to choose.**
A free-form sentence — *"the door is closed in the initial frames and then opens"* —
has no options to be biased by, and the choosing can be done afterwards in code
where it is deterministic. Approach 3's `parse_state` exists because Approach 2
proved a model could not be trusted with that step.

---

## 9 — Approach 3: caption, parse, derive

**Why.** If the model can perceive but cannot reason about time, give it only the
perception and do the temporal reasoning in code.

**Method.** Poll the clip on a 1-second grid. At each step ask *"what state is the
door in, and does it change across these frames?"* Parse the answer with string
matching — deterministic, not a model call. Derive the event from the transition.

A model was tried for the parsing step first and **scored exactly 0.00 on every
description** once option order was averaged out — the signature of answering by
position rather than reading. String matching cannot acquire that bias.

**Result.** On windows placed around known labels, 7 of 10 events hit. On
`admin.G326`, where Approach 1 reported the door opening at 90–102 s against a
3.0–5.7 s label, this returns **3.50–8.50 s, tIoU 0.406**.

**What changed.** It became the second strategy behind the same `find_events`,
switchable with `--strategy states`, so both remain runnable for comparison.

---

## 10 — What happens over a whole clip, not a window around the answer?

**Why.** Every Approach 3 number so far came from short windows placed around a
known label. That is not the task.

**Method.** Full 120 s of `admin.G326`, 1 s grid. 119 calls, 467 s.

**Result.** The timeline is almost entirely constant — closed 0–3, open 4–8, closed
9–93, open 94–96, closed after — so the grid spent nearly its whole budget
confirming that nothing had happened. 18 of 119 polls (15%) asserted no state at
all and were correctly recorded as uninformative rather than guessed.

**It also found a door opening at 94–96 s that our labels do not contain**, checked
by hand and **real**. MEVA's annotation file for the whole five-minute source holds
exactly two activity instances, both already labelled. So MEVA annotates *selected*
instances, not everything that happens. `precision@0.5` is the only metric this
reaches, since it divides by prediction count.

**What changed.** Cost became a feasibility problem, not an optimisation: at 3.92
s/call, the full eval was 9.3 hours.

---

## 11 — Can we poll only where something changes?

**Why.** If most polls confirm stillness, spend them where the picture moves. This
is the real-time design: a cheap signal decides when the expensive model runs.

**Method.** Inter-frame pixel change at 2 fps (`motion.py`), take the top-12 peaks
with a 2 s minimum gap, caption only there. `--trigger`.

**Result, in one server session, against uniform:**

| | calls | time | event 1 | tIoU |
|---|---|---|---|---|
| uniform, 1 s grid | 119 | 467 s | 3.50–8.50 @1.0 | **0.406** |
| triggered, top-12 | 12 | 68 s | 6.25–9.50 @0.4, partial | **0.000** |

**9.9× fewer calls, and it loses the event.** Nine of the twelve calls landed inside
the two real events, so the signal found the right *places*. It then could not
resolve the *moment*: the uniform grid first reads *open* at t=4.0 s, and
`trigger_min_gap_s` forbids two polls closer than 2 s — including at the strongest
peak, which it had correctly identified.

**What changed.** `trigger` is **off by default**. The result argues for triggering
to locate brackets and then sweeping densely inside them — not for triggering alone.
That two-stage variant is not built.

**It also exposed a bug the saving had been hiding.** The derivation placed each
transition at the midpoint of its bracket — correct on a uniform grid, nonsense
across an 82-second gap. A 3-second door was reported as a **42-second event at
confidence 1.0**. Fixed by refusing to interpolate past `carry_steps × step_s` and
marking such spans partial.

---

## 12 — Is any of this reproducible run to run?

**Why.** A poll's answer changed between two runs at temperature 0. If results move
on their own, no comparison above means anything.

**Method.** Three tests, in order: five identical repeats; a full server restart;
and the same timestamp under two different request histories in one session.

**Result.**

1. **Within one server instance, five repeats are bit-identical.** Whatever varies,
   it is not per-request sampling.
2. **Across a restart, the answer changes.** The poll at t=5.0 s went
   `closed → open → closed`.
3. **Within one instance, identical frames give different answers under different
   request histories.** The triggered run read t=5.0 as `closed`; the uniform run
   sixty seconds later read it as `open`. Both sample `[t, t+span_s]` through the
   same code path, so the frames were byte-identical; one had issued 4 prior
   requests, the other 5.

Most likely vLLM's batching and prefix-cache state changing floating-point
reduction order, which flips only near-ties — and t=5.0 is a near-tie, the door
panel begins to swing there. **Hypothesis, not measurement**; confirming it needs
logprobs at that poll.

**What changed.** A number here is a property of *(input, code, server instance,
request history)*, not of *(input, code)*. Comparisons are only valid inside one
server session, and the uniform-vs-triggered table above was re-run for that reason.
An earlier revision of the docs compared across instances and stated the result
confidently; that claim was wrong and was withdrawn.

---

## 13 — Do overlapping poll windows bias the boundaries?

**Why.** A poll recorded at `t` is sampled from `[t, t+span_s]`, and the parser takes
the *last* state mentioned — so a window where the state changes reports its
**ending** state. With `step_s=1.0` and `span_s=2.0` those conventions are 2 seconds
apart.

**Method.** Re-run uniform with `span_s=1.0` so windows tile instead of overlapping.
Same server session. `SPAN_S=1.0`.

**Prediction, stated before the run:** the boundary moves later and tIoU falls.

**Result. The prediction was wrong** — the interval did not move at all.

| | event 1 | tIoU | extra events | calls | time | `None` polls |
|---|---|---|---|---|---|---|
| `span_s=2.0` | 3.50–8.50 | 0.406 | — | 119 | 467 s | 18 |
| `span_s=1.0` | 3.50–8.50 | 0.406 | **71.50–72.50 @1.0** | 120 | 433 s | 5 |

The model reads t=3.0 as `closed` and t=4.0 as `open` under both settings, so the
bracket never changes. The ambiguity is real in principle and does not bite here;
**the experiment does not separate the two possibilities**, and the question stays
open rather than resolved.

Two side effects worth keeping: shorter windows made the model far more decisive
(`None` fell from 18 to 5) and 7% faster — and produced a **spurious event from a
single poll**, at confidence 1.0.

**What changed.** `span_s` stays 2.0. The single-poll event is the third appearance
of one failure: confidence measures consistency and sharpness but not *how much
evidence stands behind an interval*.

---

## 14 — Making the full evaluation affordable

**Why.** Scoring Approach 3 across the set is the one thing between measured work
and a defensible claim, and at 8,568 calls it was 9.3 hours.

**Method.** Two reductions, neither touching polling density.

**Descriptions are not subjects.** The nine expressible descriptions reduce to four
subjects — `door`, `car door`, `person`, `vehicle`. Polling each description
separately sent identical frames with an identical question. Grouping is keyed on
the state *set*, so *sits down* and *stands up* share one sweep.

**Subjects need not be separate calls.** `shared_caption` asks about every subject
in one call per timestep, one labelled line each, parsed separately.

| | calls | wall-clock | resolution | risk |
|---|---|---|---|---|
| one sweep per description | 8,568 | 9.3 h | full | — |
| grouped by state set | 3,808 | **4.1 h** | full | none |
| + shared caption | 952 | **~1.0–1.5 h** | full | attribution |

Shared captioning is **off by default**: *"the car door is open and the building
door is closed"* contains both answers, and a last-mention parse over the whole text
would give both subjects the same state. Mitigated structurally — one labelled line
per subject, longest-subject-first matching, and a subject with no line reported as
**no answer** rather than inheriting a neighbour's.

**Grouping works.** Verified against a fake captioning backend: 5 descriptions
collapse to 2 sweeps, the three door descriptions return identical intervals, and
the reversed sit/stand pair correctly returns opposite intervals from the same
polls.

### Shared captioning was measured, and it fails

Run on `admin.G326` with all four subjects, 119 calls, 1,215 s.

| subject | polls parsing to a state | per-subject baseline |
|---|---|---|
| `door` | 35 / 119 = **29%** | 101 / 119 = **85%** |
| `car door` | 7 / 119 = 6% | — |
| `person` | **0 / 119** | — |
| `vehicle` | 1 / 119 = 1% | — |

Asked about four things at once, the model mentions one or none. `person` never
parsed in 119 calls.

It is also **not fast**: 10.21 s per call against 3.92 s for a single-subject call,
nearly triple, because the answer is four times longer. The net saving over the
grouped path is ~1.5x, not 4x — 2.7 h against 4.1 h.

And the failure is worse than omission. The captions show the subjects actively
**interfering**:

> *"The car door — wait, maybe the door is a car door? Wait, the frames show a
> do…"*
> *"- car door: if the door is a car door, then the state is open, and no change."*

Naming `door` and `car door` in one prompt made the model conflate them. It
reported **"a vehicle door opens" as 5.0–97.0 s** on a clip with no vehicle-door
label, and the door event itself degraded from 3.50–8.50 (tIoU 0.406) to
4.50–9.00 (tIoU ~0.21).

**Verdict: `shared_caption` stays off.** The measurement is kept as a clean
negative result about the caption-once-query-many pattern at this model size — it
is the architecture NVIDIA's VSS uses, and at 4B with similar subjects in one
prompt it does not survive.

**What changed.** That 92-second false positive at confidence 1.0 is what forced
the **coverage** factor into the confidence score: four polls held a span that was
93% unobserved, and both existing factors were maximal on it.

---

## Not run

Stated plainly, because an unrun experiment quoted as a result is the worst kind of
error.

| experiment | why it matters | why not run |
|---|---|---|
| **Approach 3 scored across all 8 clips** | the one result that would make the comparison defensible | the harness could not run it until recently; the run is 1.5–4 h |
| `eval-control` — activity name burned into the frame | separates "cannot recognise events" from "cannot see at this resolution" | not built out |
| Qwen3-VL-8B vs Cosmos-Reason2-8B | isolates NVIDIA's post-training, same architecture and size | neither fits in 22 GiB with a 48-frame window |
| held-out set | every clip that produced a number also shaped a prompt or threshold | needs more labelled clips, not a post-hoc split |
| deriving state pairs from the description | the last human step in the pipeline | one cheap text call; unbuilt |
| logprobs at a near-tie poll | would confirm the determinism hypothesis | not instrumented |
