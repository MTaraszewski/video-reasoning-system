# Datasets — the evaluation set

What footage the evaluation runs on, why each source was chosen or rejected, and
under what licence. Working notes live in [`PLAN.md`](PLAN.md); the metric design
lives in [`DESIGN.md`](DESIGN.md).

Status vocabulary, per the `model-facts` skill: `?` unknown · `D` documented,
untested here · `PASS`/`FAIL` measured here · `CONFLICT` sources disagree.

---

## 1. What the assignment requires

> "Assemble a small evaluation set: **five to ten clips, each one to three
> minutes**, with the events you care about **labelled by hand** with start and end
> times. **Any footage you have the right to use is fine, including public
> datasets.**"

Three constraints follow, and the third is the one that eliminates most candidates.

| Constraint | Consequence |
|---|---|
| 5–10 clips, 1–3 min | Long enough to force windowing, short enough to hand-label and to keep GPU cost sane |
| Labelled by hand | Existing dataset annotations do not satisfy this. They can *cross-check* our labels, not replace them |
| Right to use | Must survive being referenced from a public repo — and must not require the reviewer to sign anything |

## 2. What "usable" actually means here

A licence question that is easy to get wrong: **we do not have to redistribute the
video.** A dataset is usable if we can ship

- **our hand labels** — always ours, always shippable, and
- **a download script** that fetches the footage from its own source.

That widens the field from "redistributable" to "publicly fetchable without an
agreement". But the assignment's other bar pulls the opposite way — it must run on
the reviewer's machine in minutes — so a source requiring a 470 GB pull, a login,
or a signed agreement fails on ease-of-use even when the licence permits it.

**The working rule:** small enough to commit → commit it. Otherwise → fetch script
plus committed labels. Never → anything requiring an account or an agreement.

---

## 2b. The brief specifies no dataset — and that shapes the selection

Verbatim: *"with **the events you care about** labelled by hand"* and *"**Any
footage you have the right to use is fine**, including public datasets."*

No dataset is named. The three phrases the brief quotes — "a person enters through
the door", "a forklift reverses", "the machine stops moving" — are illustrations of
**plain-language descriptions**, not a required event taxonomy. Building the eval
set to match those three would be reading the examples as a spec.

The selection criterion comes from the next sentence instead: *"we are interested to
find out **where the limits of their abilities are**."*

**So the unit of selection is the clip, not the dataset.** The budget is 5-10 clips.
Each is chosen because it stresses a *different failure axis*, drawn from whichever
source has the best example of that axis. Using more datasets costs nothing extra —
only clip count drives GPU cost, and the same clips run against every model under
comparison.

### The failure axes

Derived from the design's own predictions and from Roboflow's published Cosmos 3
findings, which name fast motion, small similar objects and overhead views as weak
points.

| Axis | Why it breaks things | Best source |
|---|---|---|
| **Boundary precision** | Sampling rate sets a hard floor; this measures it | Synthetic |
| **Short events** | Shorter than the sampling interval, falls between frames | Synthetic, anomaly sets |
| **Long events** | Spans many windows, stresses the merge | Synthetic, MEVA |
| **Fast motion** | Roboflow found Cosmos 3 struggles | VANTAGE, `supervision` vehicles |
| **Small / similar objects** | Roboflow found Cosmos 3 struggles | VANTAGE, Assembly101 |
| **Absence of motion** | "The machine stops" — harder than distinctive motion | `supervision` milk-bottling-plant |
| **Crowds / distractors** | Visually similar non-events inflate false positives | `supervision` subway, market-square |
| **Overhead camera** | Roboflow reported reliable failure on an overhead view | Assembly101 |
| **Negative clips** | Query genuinely absent — measures false positives | Any |

**Synthetic footage is a first-class instrument here, not a fallback.** We control
event duration, speed, size, contrast and count exactly, so any axis can be probed
in isolation with zero labelling error. Where a real dataset lacks an axis, we
construct it.

## 3. Candidates assessed

| Source | Fixed camera | Temporal annotations | Licence | Redistributable | Verdict |
|---|---|---|---|---|---|
| **MEVA** | yes | yes, 37 activity classes | **CC BY 4.0** | yes, with attribution | **Selected** |
| **Roboflow `supervision` assets** | yes | no | **unstated** | no | **Selected, fetch-only** |
| Synthetic (ours) | n/a | exact by construction | ours | yes | **Selected** |
| **VANTAGE-Bench** | yes | yes | `nvidia-evaluation-data-license` | no; gated download | **Selected**, supplementary tier |
| **ComplexVAD** | yes | yes, anomaly spans | **CC BY-SA 4.0** `D` | yes, share-alike | **Candidate** |
| **Street Scene** | yes | yes, per-frame tracks | **CC BY-SA 4.0** `D` | yes, share-alike | **Candidate** |
| **Assembly101** | yes, multi-view + overhead | yes, 1M fine-grained segments | **CC BY-NC 4.0** `D` | non-commercial only | **Candidate** |
| EPIC-KITCHENS | no, egocentric | yes, 90K segments | CC BY-NC 4.0 `D` | non-commercial only | Wrong camera |
| Roboflow Universe | — | no — image object-detection | varies | — | Does not fit |
| VIRAT | yes | yes | "VIRAT Video Dataset Protection Agreement" | no | Ruled out |
| Pexels / Pixabay / Videvo | rarely | no | CC0-ish, with caveats | disputed | Weak |
| THUMOS / ActivityNet | no | yes | per-video, unclear | no | Wrong domain |

---

## 4. Selected sources

Six sources, grouped by **how reproducible they are for the reviewer** — which is
the constraint that matters, since the brief is strict about running first-try.

### Core — anyone can reproduce, no account, no agreement

| Source | Ground truth | What ships | Answers |
|---|---|---|---|
| **Synthetic** | **Exact**, zero labelling error | Generator + clips + labels, committed | What is the model's *precision floor*, with labelling error removed? |
| **MEVA** — CC BY 4.0 | Hand-labelled, cross-checked against MEVA's own annotations | Fetch script + our labels | Does it work on real fixed-camera footage? |
| **ComplexVAD / Street Scene** — CC BY-SA 4.0 | Hand-labelled | Fetch script + our labels | Short, rare events on a static camera |

### Convenience — fetchable, licence unstated

| Source | Ground truth | What ships | Answers |
|---|---|---|---|
| **Roboflow `supervision` assets** | Hand-labelled | Fetch script + our labels, **never the video** | Absence-of-motion events and crowded scenes |

### Supplementary — requires the reviewer to accept terms

| Source | Ground truth | What ships | Answers |
|---|---|---|---|
| **VANTAGE-Bench** — eval-only, gated | Hand-labelled by us; NVIDIA's is withheld | Fetch script + our labels | Warehouse / transportation domain: fast motion, small similar objects |
| **Assembly101** — CC BY-NC 4.0 | Hand-labelled | Fetch script + our labels | **Overhead camera**, fine-grained short actions |

**Why the synthetic tier is not optional.** The capability probe measures how
precisely the model can place an event boundary. That number is meaningless if the
reference labels are themselves fuzzy by ±0.3 s. Only constructed ground truth
separates the model's error from the labeller's — and every other result is read
against it.

**Why the grouping matters more than the count.** The evaluation must stand on the
core group alone. Everything above it adds coverage; nothing above it is load-
bearing. A reviewer who accepts no terms and fetches nothing gated still gets a
complete, reproducible result.

---

## 5. MEVA — the primary real source

**Multiview Extended Video with Activities**, Kitware, collected for the IARPA DIVA
programme.
— <https://mevadata.org/> · <https://registry.opendata.aws/mevadata/>

| | |
|---|---|
| Licence | **CC BY 4.0** — `D`, confirmed by Kitware and the AWS Registry of Open Data |
| Access | **No login, no agreement.** `aws s3 sync s3://mevadata-public-01/<drop> . --no-sign-request` — no-cost via the AWS Public Dataset Program |
| Size | 328 h ground camera, 4001 clips, ~470 GB total. **We take a narrow slice** |
| Clip length | ~5 min average (330 h / 4001 clips). **Trim to 1–3 min** — CC BY permits derivatives |
| Camera | Fixed, indoor and outdoor, Muscatatuck Urban Training Center |
| Annotations | 37 DIVA activity classes, in a public GitLab repository |

**Why it fits the brief so directly.** The activity classes include
**`person_opens_facility_door`** — effectively the brief's own example, *"a person
enters through the door"*. Also `vehicle_picks_up_person`, `person_reads_document`
and 34 others, giving a range of event types from distinctive motion to subtle
state change.

**Attribution obligation.** CC BY 4.0 requires credit. Any clip or derivative we
ship must carry the attribution, and the repo must state it. This is a real
requirement, not a formality.

**Cross-checking, not substituting.** The brief asks for hand labels, so we label
by hand. MEVA's own annotations are then used to *measure our labelling error* —
which is itself worth reporting, since it bounds how seriously any tIoU difference
below that margin should be taken.

`UNVERIFIED`: whether MEVA's annotation format carries explicit start/end frame
times per activity instance, and which S3 drop is the smallest useful slice. Both
settle on first download.

---

## 6. Roboflow `supervision` assets — fetch-only

Seven sample videos shipped with Roboflow's `supervision` library, downloadable via
`supervision.assets.download_assets(VideoAssets.<NAME>)`, hosted at
`media.roboflow.com/supervision/video-examples/`.
— <https://supervision.roboflow.com/assets/>

`VEHICLES` · `VEHICLES_2` · **`MILK_BOTTLING_PLANT`** · `GROCERY_STORE` · `SUBWAY` ·
`MARKET_SQUARE` · `PEOPLE_WALKING`

**Why they earn a place.** `MILK_BOTTLING_PLANT` is close to the brief's *"the
machine stops moving"* — a machine-state event, which is a different and harder
detection problem than distinctive motion, and one MEVA does not cover.
`GROCERY_STORE`, `SUBWAY` and `MARKET_SQUARE` add crowded fixed-camera scenes.

**Licence is the caveat.** `?` — **no licence is stated**. The documentation page
carries only *"Roboflow 2023. All rights reserved."*

Handling, until that changes:

- **Fetch at runtime via the `supervision` package. Never commit the video files.**
- Our labels are our own work and ship without issue.
- Treat as a convenience tier: valuable, but the evaluation must stand without it.

Worth asking Roboflow directly — it costs nothing and the answer may simply be
"they are ours, use them."

---

## 7. VANTAGE-Bench — assessed in detail, accepted as supplementary

**VANTAGE-Bench** — the closest match on paper, and the call was re-examined
because the first version of this section gave a reason that does not hold.
— <https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench>

Fixed-camera warehouse, transportation and smart-space footage, with a
`temporal_localization/` split defined as *"predict precise start and end
timestamps for a queried event"* — our exact task, on our exact domain.

Verbatim from the dataset card:

> "This dataset is for evaluation purposes only."
> "Ground truth annotations are not publicly released. All evaluation is performed
> server-side."

Downloading **is** permitted. From the dataset page:

> "**When downloaded or used** in accordance with our terms of service, developers
> should work with their internal developer teams to ensure this dataset meets
> requirements for the relevant industry and use case and addresses unforeseen
> product misuse."

That phrasing anticipates download, and resolves the earlier `CONFLICT` about
availability. Assessing each objection against our **actual** use:

| Objection | Holds? |
|---|---|
| Ground truth is withheld | **No.** The brief requires us to hand-label anyway. We never needed NVIDIA's labels. Retracted |
| Not downloadable | **No.** Retracted — the terms explicitly contemplate download |
| "Evaluation purposes only" | **No, and it may point the other way.** Our use *is* evaluation, of an NVIDIA model. Licences of this shape grant use "solely to evaluate and test NVIDIA technologies" — which is exactly what we would be doing |
| Dataset is **gated** | **Partly.** A reviewer must hold an account and accept terms before reproducing. Real friction against the first-try bar, though modest for an ML team |
| No redistribution of clips | **Yes.** No footage in a public repo |
| Publishing **our labels** for it | **Grey area.** Timestamps and plain-language descriptions are our own work, and facts about a video are generally not the video's copyright — but this is not settled, and we are not lawyers |

**What it would add that MEVA does not: domain.** VANTAGE is warehouse,
transportation and smart spaces. MEVA is a training facility — people, vehicles,
doors. VANTAGE is the closer match to the brief's own *"a forklift reverses"*, and
would close the forklift gap noted in §10.

**What it costs.** It cannot sit in the reproducible core: gated download, no
shippable footage, and label publication that is legally unclear. Used at all, it
would be a clearly-labelled supplementary tier — our numbers reported, the data not
shipped, reproduction requiring the reviewer to accept NVIDIA's terms.

**Status: accepted as a supplementary tier.** The original rejection rested on
three reasons and only the weakest survived. It is in for the domain coverage —
warehouse and transportation footage nothing else provides — but never in the core
reproducible group.

`UNVERIFIED`: the dataset's own `LICENSE.md` has not been read. The
"solely to evaluate and test NVIDIA technologies" wording comes from NVIDIA's
similarly-named evaluation-licence family, **not** from this dataset's exact text.
That file settles both the redistribution question and the status of our labels,
and must be read before any decision to use it.

It remains useful as **prior art for the metric**: it confirms mIoU-style temporal
overlap, with Precision@0.5 secondary, is the accepted measure for this exact task
on this exact domain.

## 8. Rejected

**VIRAT** — strong domain fit, but distributed under the *VIRAT Video Dataset
Protection Agreement*. An agreement the reviewer would have to accept fails the
ease-of-use bar.

**Roboflow Universe** — image object-detection datasets. No untrimmed video with
temporal spans, which is the whole requirement.

**Pexels / Pixabay / Videvo** — CC0-style licences, but Pexels explicitly bars
redistribution on other platforms, and Videvo's attribution licence bars
redistributing unmodified video. Compounding that, stock footage is mostly not
fixed-camera operational video, so it is a poor domain match regardless.

**THUMOS / ActivityNet / Charades-STA** — the standard temporal-grounding academic
sets. Rejected on domain: YouTube footage, moving cameras, per-video rights that
are unclear at best. The system targets fixed cameras.

---

## 9. Labelling protocol

To be fixed before labelling starts, so labels stay comparable:

- Every label is `{video, description, start_s, end_s}`, times in seconds.
- **Descriptions are written as a client would phrase them** — plain language, as in
  the brief's examples — not as dataset class names. The system is open-vocabulary;
  labelling with `person_opens_facility_door` would test the wrong thing.
- Boundary convention stated explicitly and applied consistently: when an event
  starts and ends is a judgement call, and an unstated convention makes tIoU
  unreadable.
- Event types deliberately span the range the brief names: distinctive motion, state
  change, and absence of motion.
- Distractors are included on purpose — clips where a query does *not* occur, to
  measure false positives. Recall alone would flatter the system.

---

## 10. Open items

- [ ] Detail sections for ComplexVAD, Street Scene and Assembly101 (§4 lists them;
      only MEVA, `supervision` and VANTAGE have full write-ups)
- [ ] Which MEVA S3 drop is the smallest slice yielding 5–10 good clips
- [ ] Confirm MEVA annotation format carries per-instance start/end times
- [ ] Ask Roboflow about the `supervision` asset licence
- [ ] Nothing in the set has **forklifts**, one of the brief's own examples. MEVA is
      a training facility — people, vehicles, doors. Needs a separate hunt if
      warehouse material matters
- [ ] Fix the boundary convention before labelling
- [ ] Decide how the 5–10 clip budget is split across the six sources
