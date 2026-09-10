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
| **Low source resolution** | Below the model's own input size, so detail is upscaled rather than downscaled | MEVA `G474` — see §5 |

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

### What the fetched slice actually contains

Twelve clips from `drops-123-r13`, verified with `ffprobe` rather than trusted from
filenames. **Every clip's real duration matches the duration encoded in its
filename**, which is what the selection rule relies on.

| Property | 11 clips | **`G474`** |
|---|---|---|
| Resolution | 1920x1080 / 1920x1072 | **352x240** |
| Frame rate | 30 fps | 30 fps |
| Duration | 236-300 s | 299 s |
| Size | 56-203 MB | **2.2 MB** |

**`G474` is kept on purpose, as its own axis.** It is not corrupt — it is a
low-resolution camera. At 352x240 it sits *below* the 640x360 we downscale to for
the model, so its frames are upscaled and carry none of the detail the others have.

Real deployments have mixed camera quality, and "does the model still ground events
at 352x240?" is a limits question the brief explicitly asks for. It is free
evidence about a variable nothing else in the set tests.

**It must never be averaged in with the 1080p clips.** Pooling them would confound
resolution with every other difference between cameras. It is labelled and reported
as the `low_resolution` axis, separately.

### Two consequences for the eval set

- **These clips are ~5 minutes; the brief asks for 1-3.** They need trimming. This
  is also a cost lever: at 4 fps with 12 s windows and 9 s stride, a 300 s clip is
  ~34 windows *per query*. Trimming to 2 minutes cuts model calls by about 60%.
- **Two frame heights appear** — 1920x1080 and 1920x1072. A non-standard crop on
  some cameras. Harmless, but nothing may assume a fixed aspect ratio.

### MEVA's curated example clips carry their answers in the picture

MEVA publishes 121 pre-cut clips named for the activity they contain
(`ex013-enter-vehicle.mp4`), spanning ~36 activity types. They looked ideal:
guaranteed events, tiny download, and several map onto the brief's own examples —
`vehicle-reversing` for *"a forklift reverses"*, `open-facility-door` for *"a
person enters through the door"*.

**They cannot be used as evaluation data.** The clips are rendered with MEVA's
annotations burned in: a red box around the actor labelled with the activity name,
and a header giving the source file and frame number. A model can read the answer
off the frame.

They are kept as a **`positive_control` ceiling test** instead — if the model
cannot find an event whose name is written on screen, it will not find it on clean
footage. Handling, guarantees and the verification are in
[`DESIGN.md §9a`](DESIGN.md).

**How this was found:** by rendering a contact sheet and looking at it. No
automated check would have caught it — nothing in a pipeline can notice that a
dataset has written its labels into its own pixels.

### Selecting by activity — what it produced

`make data-meva MEVA_MODE=annotated` reads MEVA's `.activities.yml` files, keeps
clips whose annotations declare an activity, downloads the **original clean
footage**, and stores the annotation alongside — used to locate events, never shown
to a model.

Six clips, and the declared activities cover the brief's own examples:

| Clip | Declared activities |
|---|---|
| `admin.G326` | Enter_Facility, **Open_Facility_Door** |
| `admin.G329` | Enter_Facility |
| `bus.G331` | Object_Transfer, Purchasing, Read_Document, Text_On_Phone |
| `school.G328` | Enter_Vehicle, Open_Vehicle_Door, **Vehicle_Reversing**, Vehicle_Starting, **Vehicle_Stopping** |
| `school.G336` | Vehicle_Stopping, Vehicle_Turning_Left, Vehicle_Turning_Right |
| `hospital.G301` | Exit_Facility, Open_Facility_Door |

`Vehicle_Reversing` is the same event *shape* as the brief's "a forklift reverses";
`Vehicle_Stopping` matches "the machine stops moving"; `Open_Facility_Door` and
`Enter_Facility` match "a person enters through the door". Event lengths run from
1.0 s to 23.9 s, so short and long events are both tested on real footage rather
than only in synthetic clips.

**Two bugs found getting there.** MEVA's video filenames carry a release suffix its
annotation filenames do not (`...G329.r13.avi` versus `...G329.activities.yml`), so
every constructed URL 404'd. Worse, the errors were swallowed and the script
**exited 0 having downloaded nothing** — reporting success for an empty directory.
It now counts what it fetched and fails loudly with the likely cause.

### Annotations locate events; hands label them

`make meva-plan` parses each annotation into activity spans in seconds, chooses a
trim window, and emits a worksheet with times pre-filled and phrased as a client
would say them. Every entry carries `confirmed_by_hand: false` until a human checks
it against the contact sheet.

The brief requires hand labels, and that is what these become — but hand-labelling
is *searching* plus *judging*, and only the judging needs a person. MEVA already
knows where its activities are.

**The trim window is computed, not fixed at zero.** One clip declares events at
45 s, 64 s, 74 s — and 268 s. Trimming naively from the start would silently discard
the last, and nothing downstream would notice a labelled event had been cut away.
The chosen window maximises *wholly contained* events and reports what it drops.

### Screening for activity, and where it fails

`make screen-clips` measures the fraction of pixels changing between samples, as an
objective answer to "does anything happen here" — no filenames, no annotations, no
interpretation. It exists because two rounds of clip selection produced unusable
footage discovered only after downloading and inspecting by hand.

**It has four known blind spots**, all measured against clips whose content is known
by construction:

| Blind spot | Evidence |
|---|---|
| Slow events read *faint* | a 27 s traverse scored 1.07 % |
| Short events read *static* | a 0.15 s event scored 0.07 % at 2 fps |
| "Active" does not mean "contains your event" | a negative clip scored 3.42 % |
| Only samples the first N seconds | a clip whose event sits at 268 s read *static* |

Compounding that, **fixed-camera events are small in frame**: a person opening a
door 40 m away moves a fraction of a percent of pixels. The thresholds were
calibrated on synthetic clips where a box crosses the whole frame, and are too
blunt for real surveillance.

So the screen is a filter for *completely dead* footage — which is what it caught,
twice — and the annotations are the better guide to what is worth labelling.

### The caveat about this slice — and what it cost

All twelve clips first fetched were the **same five-minute window seen from twelve
different cameras** at one location. This section previously ended: *"Sampling
across dates and times would fix it, and should be settled before labelling
begins."*

It was not settled before labelling began. The cost was one full labelling pass:
of fourteen candidates, **eight were unlabellable and one clip lost every event it
had**, leaving three confirmed events across two cameras pointed at the same
building. Not an eval set. The diagnosis is below, and it turned out to have
nothing to do with sampling dates.

### Actor size decides whether an event can be labelled at all

The clips were selected because their annotation **declared an activity**. Nothing
asked how big the actor was in frame — and the annotation is silent on it, so the
question never came up until contact sheets made it unavoidable.

MEVA publishes `.geom.yml` beside `.activities.yml`: per-frame bounding boxes for
every actor. The first fetch discarded them, filtering the S3 listing with
`awk '/activities\.yml$/'`. They answer the size question directly, at 5–70 KB per
clip against 56–203 MB per video — so the whole corpus can be screened for less
than the cost of downloading one clip.

Measuring median actor height during each declared event, against verdicts reached
by eye **before this measurement existed**:

| Camera | Median actor h | Hand verdict |
|---|---|---|
| `G326` admin | **694 px** | two events confirmed |
| `G329` admin | **295 px** | confirmed (needed zooming) |
| `G331` bus | **267 px** | suspect, could not verify |
| `G301` hospital | **121 px** | all rejected |
| `G328` school | **41 px** | all rejected |
| `G336` school | **38 px** | all rejected |

The ranking is **monotonic with the hand verdicts across all six clips**. The
threshold sits between 121 and 267 px; `scripts/screen_geom.py` bands at 250 and
120 px and is calibrated on this table rather than on a guess.

**The screen ranks; it does not reject.** A false positive costs ten seconds
looking at a sheet. A false negative is silent — the clip never appears and nothing
records that it was dropped. `G329` was nearly lost that way to an eyeball
judgement, so every clip stays in the output and the band only annotates.

Applied to the full corpus — 64 clips across six days — it returns **17 in the
"good" band**, against the brief's five to ten. It also found `Vehicle_Reversing`
at **232 px on `school.G300`**, the same event that was unlabellable at 41 px on
`G328`. The brief's "a forklift reverses" was recoverable all along; we had simply
picked the wrong camera.

**The real lesson is not about dates.** Sampling across days was never the fix.
Selecting per *event* rather than per *clip* was: `G336` appears at 204 px for
`Open_Trunk` and 38 px for `Vehicle_Stopping` — the camera is not uniformly bad, we
had picked its worst events.

### Sheets locate; zoom adjudicates

Two candidate verdicts read off contact sheets were overturned by zooming into the
source, and both times the sheet reading was **biased late on the start**:

| Event | Read from the sheet | After zoom |
|---|---|---|
| `G329` "enters through the door" | "no door visible, person on a staircase" | the person does go through doors |
| `G326` "opens a building door" | "door shut at 4.2 and 4.7, so the 3.0 start is 1.8 s too early" | person visible behind the glass working the knob from 3.0 s — **MEVA is right** |

One cause: a sheet shows the **object** changing state — the door panel swinging —
well after the **actor** began the act. Reaching for a knob is small, often behind
glass, and survives neither a 3.4x downscale nor 0.5 s sampling. Nothing about
those frames looks ambiguous, which is what makes the error dangerous.

Consequences, both acted on:

- **Event sheets now size their tiles from the measured actor height**, targeting
  ~150 px on the sheet regardless of camera distance — a 232 px actor gets a
  1241 px tile, a 694 px actor gets 480 px.
- **MEVA's spans were more right than assumed.** Of three "too wide" starts
  flagged, the two checked at full resolution were correct. The genuine over-wide
  spans were a different failure — whole-trajectory tracking, 110 s for a drop-off
  and 24 s for coming out of a door — and those *are* obvious on a sheet, precisely
  because they span the entire clip.

### The eval set as labelled

**8 clips, 120 s each, 15 hand-confirmed events**, in `data/eval/labels.json`.

| Clip | Events |
|---|---|
| `03-07 admin.G329` | a person enters through the door |
| `03-07 admin.G326` | opens a building door; enters through the door |
| `03-07 bus.G340` | a person gets into a vehicle |
| `03-11 school.G300` | vehicle door opens; **a vehicle stops moving**; drops someone off; gets out of a vehicle |
| `03-12 school.G423` | sits down; stands up; hands an object to another person; buys something |
| `03-12 admin.G326` | comes out through the door |
| `03-13 school.G300` | **a vehicle reverses** |
| `03-15 school.G421` | a person buys something |

All three of the brief's worked examples are covered — *"a person enters through
the door"*, *"a forklift reverses"* (`Vehicle_Reversing`), *"the machine stops
moving"* (`Vehicle_Stopping`). Event durations run 1.4 s to 13.6 s, median 2.3 s,
so short and long are both tested on real footage.

**Boundary times are MEVA's, deliberately.** Where a hand reading disagreed, the
annotation was kept: it is traceable to a published source, and the one case
checked at full resolution proved the annotation right and the hand reading wrong.

### Reproducing the eval set

The trimmed clips are **not committed** — 400 MB, and the source is public. What is
committed is `labels.json`, and each entry carries `source`, `trimmed_from_s` and
`duration_s`. `make data-eval` fetches the source footage from MEVA's public bucket
(no credentials) and re-cuts each clip with the same settings.

Verified rather than asserted: rebuilding `school.G300` produced a clip with the
same duration, the same dimensions, and the **same SHA-256 over 60 sampled frames**
as the original. Re-encoding, not stream copy — a stream copy cuts only at
keyframes, so the real start drifts by up to a keyframe interval and every label
silently shifts with it.

### Kept as measured negatives

`G328` (41 px), `G336` (38 px) and `G301` (121 px) stay in the repo, unlabelled and
excluded from every score. They are a result: **MEVA's distant cameras place actors
below the size at which a human can confirm an event, so there is no honest ground
truth to score a model against.** They also enable a test nothing else can run —
MEVA declares events there that no person can verify, so asking the model for them
measures whether it invents confident localisations when the evidence is absent.
Reported separately, never pooled.

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
- [x] Which MEVA S3 drop is the smallest slice yielding 5–10 good clips —
      `drops-123-r13`, screened by actor size: 17 "good" clips from 64
- [x] Confirm MEVA annotation format carries per-instance start/end times — yes,
      `timespan: [{tsr0: [frame0, frame1]}]`, and `.geom.yml` carries per-frame
      actor boxes
- [ ] Ask Roboflow about the `supervision` asset licence
- [ ] Nothing in the set has **forklifts**, one of the brief's own examples. MEVA is
      a training facility — people, vehicles, doors. Needs a separate hunt if
      warehouse material matters
- [x] Fix the boundary convention before labelling — MEVA's times are kept
      verbatim; hand readings that disagree are recorded in `note`, not applied
- [x] Decide how the 5–10 clip budget is split across the six sources — 8 real
      clips from MEVA, synthetic for the precision floor (exact ground truth)
