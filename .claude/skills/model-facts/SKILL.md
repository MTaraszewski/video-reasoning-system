---
name: model-facts
description: Verify any claim about a model, framework, dataset or licence against primary sources before it enters a document, a decision, or code. Use whenever a model id, VRAM figure, context limit, licence, serving command, benchmark score, fps recommendation or capability claim is about to be written down or relied on — and whenever a vendor, blog, benchmark or brief asserts a capability that has not been measured here.
---

# Verifying model and tooling claims

A plausible-sounding recollection is not a fact. Model ids, VRAM figures, serving
flags and capability claims are exactly the things that are remembered slightly
wrong, and a single wrong character in a repo id means nothing runs.

## When this applies

Before writing down or acting on any of:

- a model / dataset / image repo id or version
- a licence, or whether weights are gated
- VRAM, context length, max frames, supported precision
- a serving command, flag, or framework version
- an fps or sampling recommendation
- a benchmark score or ranking
- **a capability claim** — "it can localise events", "it supports video"

## The status vocabulary

Every claim carries one of these. Nothing is promoted without evidence.

| Token | Meaning |
|---|---|
| `?` | Nobody claims it. Unknown |
| `D` | Vendor-documented, **untested here** |
| `B` | Asserted by a brief / blog / third party only, no vendor doc, **untested here** |
| `PASS` / `FAIL` / `PART` | **Measured here** — always with the value and the date |
| `CONFLICT` | Two sources disagree. Record both. Do not pick one silently |

`D` and `B` are **not** verification. They record who is making the claim. Only a
measurement made here promotes a claim to `PASS`.

## Source hierarchy

Prefer, in order:

1. The model card / dataset card / official repo itself
2. The framework's own API docs or supported-models table
3. Vendor engineering blog or technical report
4. Recipes and cookbooks published by the vendor or framework
5. Third-party benchmarks and leaderboards — useful, but check the subject model
   actually appears in them
6. Marketing copy and press releases — weakest; often self-reported and
   unverifiable

Never a search-result snippet alone. Open the page.

## Rules

- **Record the URL next to the claim.** A fact without its source is a rumour with
  good posture.
- **Quote verbatim** for anything load-bearing — a capability claim, a licence
  term, a serving command. Paraphrase drifts.
- **Distinguish "documented absent" from "absent".** A model card not mentioning a
  capability is *silence*, not a denial. Record it as silence, and say what test
  would settle it.
- **A self-reported ranking is not third-party evidence.** If a vendor claims a
  leaderboard position, check the public leaderboard. If the model is not listed,
  say so.
- **Check licences for the actual intended use.** "Open weights" does not imply
  redistributable, and an evaluation-only dataset licence blocks shipping clips in
  a public repo even when the download works.
- **Report conflicts, do not resolve them by preference.** Mark `CONFLICT`, cite
  both, and say what would settle it.
- **Exact ids and versions, character for character.** Hyphenation, casing and
  version suffixes are load-bearing.

## Output

For each claim: the claim, its status token, the source URL, a verbatim quote if
load-bearing, and — when unverified — the specific test that would settle it.

Then state plainly what is still unknown. An honest `?` is worth more than a
confident guess, and the gaps are what the next experiment is for.
