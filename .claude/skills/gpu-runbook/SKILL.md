---
name: gpu-runbook
description: Working on a rented, metered GPU instance — what to do before the meter starts, the order of operations on the box, and how to capture measured numbers instead of guessed ones. Use when provisioning or driving a cloud GPU, when a step will burn GPU-hours, or when writing a VRAM, latency or cost figure into a document.
---

# Working on a metered box

Two rules underneath everything here:

1. **Anything that can be done before the meter starts, must be.**
2. **Every number that reaches a document must have been measured on the box**, not
   inferred from parameter counts or copied from another region.

Project specifics — instance, region, prices, weight sizes — live in
`RUNBOOK.md`. This is the method.

## Before renting

The instance should have nothing to do but run. Confirm all of these first:

- [ ] Every code path works locally against the stub backend
- [ ] The repo is **pushed** — the box gets it by `git clone`, never by rsync from a
      laptop, or the two silently diverge
- [ ] Datasets are staged in object storage **in the instance's region**
- [ ] The region is confirmed *from the resource itself*, not from a default that an
      environment variable can override
- [ ] Quota is confirmed as an applied value, not inferred from a console dropdown
- [ ] `first-run-check` has been run — the clone is a first run

Developing on a rented box is the most expensive way to write code.

## On the box, in order

Cheapest and most diagnostic first, so a failure costs the least:

1. **Prove the GPU is visible to containers**, not just to the host. `nvidia-smi`
   on the host passes while containers see nothing — the most common silent failure.
2. **Pull images** before they are needed.
3. **Fetch data** from same-region object storage.
4. **Serve the model, and confirm which model is actually served.** A mismatch
   between the requested and loaded model attributes every later result to the
   wrong thing, which is worse than an error because it looks like data.
5. **Validate the deployment against a published benchmark** before trusting any
   novel measurement. Wrong preprocessing or precision yields a model that runs,
   answers plausibly, and scores badly — and the week gets spent blaming the model
   for a serving bug.
6. **Only then measure** the thing you came for.

## Capturing numbers

Take these while the box is up, because they cannot be reconstructed later:

| Measure | Why it cannot wait |
|---|---|
| Peak VRAM per model | Decides instance sizing; unpublished for most models |
| Max frames or tokens per call before failure | Sets window length, and therefore the whole long-video strategy |
| Latency per call, and per video-minute | Half of any cost figure |
| Actual weight download size and time | First-run experience for anyone else |
| The exact serving command that worked | Reproducibility. Record the image digest, not the tag |

Write them down **as they are taken**, not from memory afterwards.

## Cost discipline

- Stop the instance between sessions. The meter runs on state, not on use.
- Ephemeral instance storage is wiped on stop: fine for data restorable from object
  storage, never for a weight cache.
- Stop, do not terminate, or the persistent volume goes with it.
- Prefer one long focused session to several short ones — model loading is a fixed
  cost paid each time.
- If a step is going to take a while, start it and do something else; idle
  watching is billed at the same rate.

## When something fails on the box

Resist debugging in place — that is the expensive way. Instead:

1. Capture the full error and the exact command
2. Ask whether it can be reproduced locally against the stub. If yes, **stop the
   instance** and fix it locally.
3. Only debug on the box what genuinely requires the GPU

A bug that reproduces locally should never be fixed on a metered machine.
