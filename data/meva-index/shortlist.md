# Proposed eval clips — selected by event, not by clip

Selection rule, in priority order:

1. **Cover the brief's three worked examples.** It names "a person enters through
   the door", "a forklift reverses", "the machine stops moving". A set that cannot
   test those is testing something other than what was asked for.
2. **Actor large enough to hand-label.** Calibrated on six clips judged by eye
   before the screen existed; see `labelling-notes.md`.
3. **Scene and camera diversity.** Ten clips from one camera would measure one
   camera.
4. **Multi-event clips preferred.** Same download, more labels.

| # | Clip | Best h | Events | Why this one |
|---|---|---|---|---|
| 1 | `2018-03-07.16-50-01.16-55-01.admin.G326` | 705 | Enter Facility 705, Open Facility Door 665 | **Already local, already hand-confirmed.** The brief's "person enters through the door", twice |
| 2 | `2018-03-13.17-40-08.17-45-08.school.G300` | 232 | **Vehicle Reversing 232** | **The brief's "a forklift reverses".** The single most valuable addition — G328 had this event at 41 px and it was unlabellable |
| 3 | `2018-03-11.16-20-04.16-25-04.school.G300` | 342 | Vehicle DropsOff Person 342, **Vehicle Stopping 322**, Open Vehicle Door, Exit Vehicle | **The brief's "the machine stops moving"**, plus three more. Best events-per-download in the corpus |
| 4 | `2018-03-12.10-00-01.10-05-01.school.G423` | 317 | Object Transfer 317, Purchasing 317, Stand Up 278, Sit Down 234 | Four person-object interactions. Different failure shape from doors and vehicles |
| 5 | `2018-03-07.16-55-00.17-00-00.bus.G340` | 475 | Enter Vehicle 475 | Person-vehicle interaction at large size; a different scene again |
| 6 | `2018-03-12.10-00-02.10-05-01.admin.G326` | 787 | Exit Facility 787 | Largest actor in the corpus, and *exit* complements *enter* — the same door, the opposite direction |
| 7 | `2018-03-15.14-55-00.15-00-00.school.G421` | 662 | Purchasing 662, Laptop Interaction 308 | Deliberate control: `bus.G331` Purchasing at 267 px was judged "suspect, hard to verify". The same activity at 662 px separates "too small to see" from "too subtle to see" |
| 8 | `2018-03-07.16-50-00.16-55-00.admin.G329` | 295 | Enter Facility 295 | **Already local, already hand-confirmed** (after zooming) |

Two local, six to fetch. Roughly 16 candidate events before hand-rejection.

## Coverage of the brief's examples

| Brief example | Covered by |
|---|---|
| "a person enters through the door" | #1 Enter Facility, #8 Enter Facility, #6 Exit Facility |
| "a forklift reverses" | #2 Vehicle Reversing |
| "the machine stops moving" | #3 Vehicle Stopping |

Four scenes (admin, school, bus), six cameras, indoor and outdoor.

## Kept as documented negatives — no new download

`school.G328` (41 px) and `school.G336` (38 px) are already local and stay in the
repo as a measured limit: **MEVA's distant cameras place actors below the size at
which a human can label them, so there is no honest ground truth to score a model
against.** Discovered by trying to label them, which is why they are worth keeping.

## One assumption stated

Sizes are pixels at each clip's own source resolution, and resolution is not known
until download; MEVA's slice contains at least one 352x240 camera. The bias is
conservative — a low-resolution source yields small boxes and ranks low, so it
would be passed over rather than wrongly selected. Worth confirming with `ffprobe`
after fetch, not before.
