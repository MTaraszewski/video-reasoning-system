---
name: first-run-check
description: Simulate a reviewer running this repo cold — a clean clone on a machine that has never seen it, following only the README, reading no source. Use before pushing anything meant to be run by someone else, before cloning onto a rented GPU box, and whenever tempted to claim something "works".
---

# Running it the way a stranger will

Every local run has warm caches, a built image, files that exist because an earlier
command made them, and bugs fixed between attempts. None of that is true for the
person who clones the repo. The gap between "it works here" and "it works there" is
where the expensive failures live — and on a metered GPU box, that gap is billed.

## When to run this

- Before pushing anything a reviewer will run
- **Before cloning onto a rented instance** — the clone *is* a first run, and
  debugging it costs GPU-hours
- Whenever about to describe something as working, done, or verified

## The rules

1. **Fresh directory.** `git clone` into a path that has never held this project.
   Not a `git pull` in the working tree.
2. **Read only the README.** If a step is not written down, it does not exist. Do
   not fall back on knowledge of the code — that knowledge is the thing being
   tested.
3. **Type only what the README says**, in the order it says it.
4. **No fixing while running.** Note the failure, finish the run, then fix. Fixing
   as you go hides how many separate breakages there were.
5. **Time it.** The brief's bar is minutes. Record the wall clock to first useful
   output.

## What counts as a failure

Not just crashes:

- a command that needs an argument the README does not mention
- a file or directory assumed to exist
- an environment variable the README never names
- an error message that does not say what to do next
- a step that works but takes far longer than the README implies
- output that looks like success while doing the wrong thing

That last one is the dangerous class. An exit code of 0 is not evidence.

## Things that mask a broken first run

Check each explicitly, because each one silently makes the local machine unlike a
clean one:

| Masking factor | How it hides breakage |
|---|---|
| Built images in the local cache | Hides a broken Dockerfile or unpinned base |
| Files created by an earlier command | Hides a missing `mkdir`, or an assumed input |
| Environment variables in the shell | Hides a missing default — and can silently override one |
| Credentials or tokens already present | Hides an undocumented auth requirement |
| Data already downloaded | Hides a fetch step that fails or takes far longer than stated |
| A tool installed for other work | Hides an undeclared dependency |

## Reporting

State, separately and without merging them:

- **what was run**, verbatim, in order
- **what passed**, with the evidence
- **what failed**, with the exact output
- **wall-clock time** to first useful result
- **every step the README failed to mention**

Never write "works end to end" unless every step in that claim was executed in this
run. "Partially verified" is the honest phrase when only part was — see
[[never-claim-verified-without-running-it]] in spirit: the exit code is the
evidence, not the artifacts left behind.
