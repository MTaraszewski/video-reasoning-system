# Runbook — provisioning and running on a rented GPU

What to rent, what downloads when, and what it costs. Design rationale is in
[`DESIGN.md`](DESIGN.md); dataset licences in [`DATASETS.md`](DATASETS.md).

Figures marked `UNVERIFIED` have not been measured on hardware yet. Prices are
us-east-1 on-demand and vary by region — check before committing.

---

## 1. What downloads, and when

The single most useful thing to know before starting a metered instance: **weights
are not fetched until the model server first starts.** Pulling the container image
is a separate, earlier step.

| Step | Downloads | Size |
|---|---|---|
| `make build` | nothing external — builds the finder image locally | 2.09 GB image |
| `make pull` | the **vLLM container image** only | multi-GB |
| **`make serve`** | **the model weights, from Hugging Face, on first start** | see below |

Weights land in `HF_HOME=/weights`, which is the named Docker volume `weights`.

- They **survive** `make down`, container restarts and instance stop/start.
- They are **destroyed** by `make clean-weights` (`docker compose down -v`).
- Each `MODEL` downloads separately, so working through the model matrix pulls
  each one once.

### Approximate weight sizes

BF16, so roughly 2 bytes per parameter. `UNVERIFIED` — inferred from parameter
counts, not measured.

| Model | Params | Weights |
|---|---|---|
| `nvidia/Cosmos3-Edge` | 4B | ~9 GB |
| `nvidia/Cosmos-Reason2-2B` | 2B | ~5 GB |
| `nvidia/Cosmos-Reason2-8B` | 8B | ~18 GB |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | ~18 GB |

**All four cached ≈ 50 GB.**

---

## 2. Instance selection

The binding constraint is **Cosmos-Reason2-8B's documented minimum of 32 GB GPU
memory** ([model card](https://huggingface.co/nvidia/Cosmos-Reason2-8B)). That one
figure rules out every 24 GB card, regardless of how well the 4B primary would fit.

| Instance | GPU | VRAM | vCPU | RAM | ~$/hr | Full matrix? |
|---|---|---|---|---|---|---|
| `g6.xlarge` | L4 | 22.4 GB | 4 | 16 GB | ~$0.80 | ✗ Edge only |
| `g5.xlarge` | A10G | 24 GB | 4 | 16 GB | ~$1.01 | ✗ Edge only |
| **`g6e.xlarge`** | **L40S** | **44.7 GB** | 4 | 32 GB | **~$1.86** | ✓ |
| `g6e.2xlarge` | L40S | 44.7 GB | 8 | 64 GB | ~$2.24 | ✓ more headroom |

**Recommended: `g6e.xlarge`.**

If only `Cosmos3-Edge` is ever run, `g6.xlarge` at under half the price is enough —
4B in BF16 is ~9 GB of weights and fits 22 GB comfortably. But the moment the 8B
comparison is wanted, that instance has to be replaced, and the comparison is what
makes any single number interpretable.

`g6e.2xlarge` buys 8 vCPU instead of 4. Worth it only if CPU-side video decoding
turns out to be a bottleneck — which it should not be next to model inference.
Measure before paying for it.

---

## 3. Storage — 300 GB gp3 EBS root volume

```
vLLM image        ~15 GB
finder image        2 GB
weights (all 4)    50 GB
datasets           ~5 GB   (VANTAGE size UNVERIFIED)
OS + Docker        20 GB
                  ------
                   ~92 GB   ->  300 GB leaves real headroom
```

**Use EBS, not the instance store.** NVMe instance storage is *ephemeral*: it is
wiped when the instance stops. With a stop/start workflow across a week, 50 GB of
weights would be re-downloaded every single time. The `weights` Docker volume lives
in Docker's data root on the EBS root volume, which is what we want.

**Stop, do not terminate.** Terminating destroys the EBS volume and the weight
cache with it.

---

## 4. Sequence

Everything slow and free happens locally, before the meter starts.

| # | Where | Command | Why here |
|---|---|---|---|
| 1 | local | `make preflight && make build` | Verify the host, build the finder image |
| 2 | local | `make data` | Generate synthetic clips, fetch real ones |
| 3 | local | `make s3-push S3_BUCKET=<bucket>` | Stage clips to your own bucket |
| — | — | *rent the instance, clone the repo on it* | |
| 4 | gpu | `make preflight-gpu` | Prove containers can see the GPU |
| 5 | gpu | `make pull` | Pre-pull the server image |
| 6 | gpu | `make s3-pull S3_BUCKET=<bucket>` | Same-region fetch: seconds, not minutes |
| 7 | gpu | `make serve-bg` | Start the model; blocks until the endpoint answers |
| 8 | gpu | `make probe` | **Can the model ground events in time?** |
| 9 | gpu | `make eval DATASET=<labels.json>` | The measured run |
| 10 | gpu | `make down` | Stop containers before you stop paying |

**Why steps 2–3 are local.** Fetching MEVA from NVIDIA's public bucket takes
minutes. Doing that on a rented box bills GPU time for a download. Staging to a
bucket in the GPU's region turns step 6 into seconds. Same reasoning for step 5:
a metered instance should never idle on a multi-GB image pull.

**Steps 8 and 9 are not implemented yet.** They are the increments after the
current one; the sequence states the intended path.

---

## 5. Cost

Rough, for planning only.

| Activity | Time | Cost at ~$1.86/hr |
|---|---|---|
| First serve: image pull + weight download | ~20–40 min | ~$1 |
| Capability probe | ~1 hr | ~$2 |
| fps / window sweep | ~2 hr | ~$4 |
| Full model x clip matrix | ~3–5 hr | ~$6–10 |
| **Total, with slack** | | **~$25–40** |

Storage adds roughly $0.08/GB-month, so 300 GB gp3 is about $24/month — trivial
next to GPU time, but it accrues while the instance is stopped.

---

## 6. Two risks that can cost days

### GPU quota — check this first
New AWS accounts frequently have a **quota of zero** for *"Running On-Demand G and
VT instances"*. `g6e.xlarge` needs 4 vCPU of it. The increase is a support request
and can take **hours to days**.

This is the one blocker no amount of local preparation can shorten. Request it
before anything else.

### `UNVERIFIED` — GPU architecture support
The Cosmos-Reason2 model card lists supported microarchitectures as **Hopper and
Blackwell**. The instances above are neither:

| GPU | Architecture |
|---|---|
| L40S | Ada Lovelace |
| L4 | Ada Lovelace |
| A10G | Ampere |

BF16 is supported on Ampere and later, so it will very likely run — but NVIDIA does
not list these as supported, and AWS has no affordable single-GPU Hopper option
(`p5` instances are 8x H100). Choosing `g6e.xlarge` means accepting that risk.

**Mitigation:** serve `Cosmos-Reason2-8B` early in the first session rather than at
the end. If Ada is a problem, we find out with the rest of the session still
available to adapt, instead of discovering it after the budget is spent.
