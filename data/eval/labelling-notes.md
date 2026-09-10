# Pre-verdict on the 14 candidate events

Method: one dense sheet per event (`make event-sheets`), window +/-4s of padding,
2 fps, green border = inside the candidate window. Every sheet opened individually.
These are a SECOND opinion to check against, not labels. Nothing deleted.

| # | Sheet | Candidate | Verdict | Note |
|---|---|---|---|---|
| 1 | G301__00 opens building door | 3.0-5.7 | **can't see** | only a shadow in the recessed doorway at 1.0-2.5s, i.e. BEFORE the window; deep shade, ~20px |
| 2 | G301__01 comes out through door | 3.0-26.9 | **reject** | 24s window, scene completely static; no actor visible at any point |
| 3 | G326__00 opens building door | 3.0-5.7 | **confirm, start wrong** | door visibly SHUT at 4.2 and 4.7s, swinging open at 5.0. Suggest ~4.8-5.6 |
| 4 | G326__01 enters through door | 5.2-7.5 | **confirm** | hand on door 5.2, through doorway 6.2, inside 7.2. Suggest 5.2-7.2 |
| 5 | G328__00 vehicle door opens | 3.0-5.2 | **reject** | MT: a person IS visible moving, but no door opening. Actor visible, act not |
| 6 | G328__01 person gets into vehicle | 3.0-6.5 | **reject** | MT: person visible moving, but not entering a vehicle |
| 7 | G328__02 vehicle reverses | 22.0-29.6 | **can't see** | something shifts near the white van; reversing vs forward not separable |
| 8 | G328__03 vehicle starts moving | 32.0-34.0 | **can't see** | no discernible change across the window |
| 9 | G329__00 enters through door | 3.0-4.8 | **confirm, slightly wide** | MT first read no door, then **zoomed and confirmed the person does go through doors**. Original verdict restored. Empty at 3.0, figure at 3.5, clear 4.0-4.5. Suggest ~3.4-4.6 |
| 10 | G331__00 person buys something | 16.0-21.9 | **suspect** | group IS at the counter top-of-frame, but a transaction is not resolvable at ~40px |
| 11 | G331__01 hands object to another | 19.3-21.6 | **suspect** | MT: maybe true, hard to verify. Same counter, ~40px |
| 12 | G336__00 vehicle turns left | 3.0-9.4 | **suspect** | a dark car does traverse the far road 3.5-8.5; "left" not confirmable at ~25px |
| 13 | G336__01 vehicle stops moving | 15.9-16.9 | **can't see** | actors at the treeline, ~30px |
| 14 | G336__02 vehicle turns right | 22.2-26.8 | **can't see** | road empty throughout the window |

**Tally after MT's review and the zoom recheck: 3 confirm (`G326` x2, `G329`), 3 suspect, 8 unusable.**

MT reviewed independently and agreed on every entry not annotated `MT:` above.
Two of their corrections change the picture materially:

- **G328 events fail on the ACT, not on actor size.** A person is visible; the
  door-opening and the vehicle-entry are not. My "too small to see" was the wrong
  diagnosis for these two — the actor resolves, the fine-grained action does not.
- **G329 needed zooming to settle.** First read from the sheet: no door. After
  zooming into the source: the person does go through doors, and the candidate
  stands. The event was never the problem — **the sheet was**. Tiles are rendered
  at 560px wide from 1920px source, a 3.4x downscale, so a 295px-tall actor
  becomes ~86px on the sheet and door-frame detail disappears. A labelling aid
  that loses the detail the label depends on produces false rejections, and this
  one nearly cost us a good clip.

## Two findings that outlast this table

**1. MEVA spans are systematically wider than the visible act.** G326__00 is the
clean case: the door is shut for the first 1.7s of its own labelled window, and a
second sheet (G326__01, from the same camera) independently confirms it. Scored
as-is, tIoU against 3.0-5.7 would punish a model that answered 4.8-5.6 — the
correct answer. This is why the candidates needed confirming rather than adopting.

**2. MEVA's outdoor cameras place actors below the size a human can label.**
At 15-50px there is no boundary to read, so there is no honest ground truth to
score against. This is a property of the footage, not of our tooling: rendering
larger sheets does not add detail that the sensor never captured.

**3. Our own phrasing map asserts things the camera does not show.** `PHRASING`
turns MEVA's `Enter_Facility` into "a person enters through the door". On G329
there is no door in shot. The class name was a reasonable summary of the *activity*;
it is not a description of the *footage*. Any mapping from a taxonomy to
client language has to be checked per clip, not applied per class.

## Where this leaves the eval set

Two confirmed events, both from `G326` — a single clip. `G301`, `G328` and `G336`
contribute nothing; `G329` needs re-phrasing or dropping; `G331` is unresolved.

That does not meet the brief's "five to ten clips", and no amount of re-rendering
fixes it: the shortfall is in the footage, not the tooling.

**Root cause in our own selection.** Clips were chosen because their annotation
*declared an activity* — never because the activity was *visible at labellable
size and specificity*. `screen_clips.py` filtered on the annotation, and the
annotation is silent about how big the actor is in frame. That is the missing
criterion.

---

## Post-labelling: what the contact-sheet method actually does

The eval set finished at **8 clips, 15 hand-confirmed events**, covering all three
of the brief's worked examples. But the labelling pass produced a finding about the
method itself, and it is not a flattering one.

**Contact sheets systematically under-detect the START of an act.** Twice, a
verdict read off a sheet was overturned by zooming into the source, and both times
the sheet reading was mine and the correction was MT's:

| Event | Sheet reading | After zoom |
|---|---|---|
| `G329` "enters through the door" | "no door visible, person on a staircase" | person does go through doors; candidate stands |
| `G326` "opens a building door" | "door shut at 4.2 and 4.7, so the 3.0 start is 1.8s too early" | person visible behind the glass approaching and working the knob from 3.0s; MEVA is right |

Both failures share one cause. A sheet shows the **object** changing state — the
door panel swinging — long after the **actor** has begun the act. Reaching for a
knob is small, is often behind glass, and survives neither a 3.4x downscale nor
0.5 s sampling. So a sheet-based reading is biased LATE on starts, and confidently
so: nothing about the frames looks ambiguous, which is what makes it dangerous.

Two consequences:

1. **MEVA's spans were more right than we assumed.** Of three "too wide" starts we
   flagged, the two that were checked at full resolution were correct. The genuine
   over-wide spans were of a different kind entirely — whole-trajectory tracking
   (110 s for a drop-off, 24 s for coming out of a door), which is obvious on a
   sheet precisely because it spans the whole clip.
2. **Sheets locate; zoom adjudicates.** Sheets are a search tool, and they are good
   at it — they made 26 candidates reviewable in one sitting. They are not evidence
   about a boundary. Any boundary correction taken from a sheet alone should be
   treated as a hypothesis, which is exactly what the one we acted on turned out
   to be.

The choice to keep MEVA's candidate times rather than our own tightenings was
therefore the right one, and for a reason we only learned afterwards.
