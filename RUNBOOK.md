# Runbook — provisioning and running on a rented GPU

What to rent, what downloads when, and what it costs. Design rationale is in
[`DESIGN.md`](DESIGN.md); dataset licences in [`DATASETS.md`](DATASETS.md).

Figures marked `UNVERIFIED` have not been measured on hardware yet. Prices are
approximate US-region on-demand rates and vary — check the console before
committing. Reports of a ~20% GPU price rise are not reflected here.

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

### The constraint is the largest SINGLE model, not the sum

Models are served **one at a time** — `make serve` loads one `MODEL`, the matrix
runs, then the next is loaded. So VRAM must fit the biggest single model, never the
50 GB of weights on disk.

| Job | Largest model | VRAM needed |
|---|---|---|
| **Cosmos3-Edge only** | 4B, ~9 GB weights | **24 GB is enough** |
| **The full model matrix** | Cosmos-Reason2-8B | **≥32 GB** — NVIDIA's documented minimum |

That 32 GB figure is from the [Cosmos-Reason2-8B model
card](https://huggingface.co/nvidia/Cosmos-Reason2-8B). At BF16 the weights are only
~16 GB; the rest is KV cache, the vision tower and activations. We cap
`--max-model-len 32768` rather than the full 256 K, which should reduce KV pressure
considerably — but the documented minimum stands until measured.

### Single-GPU options

Two columns are easy to confuse: **VRAM** is memory on the GPU and is what limits
model size; **RAM** is host memory. `g7e.2xlarge` has 96 GB of the former and 64 GB
of the latter.

| Instance | GPU | Arch | VRAM | vCPU | RAM | ~$/hr | Fits ≥32 GB? |
|---|---|---|---|---|---|---|---|
| `g4dn.xlarge` | T4 | Turing | 16 GB | 4 | 16 | ~$0.53 | no |
| **`g6.xlarge`** | L4 | Ada | 24 GB | 4 | 16 | **~$0.80** | no — **Edge only** |
| `g6.2xlarge` | L4 | Ada | 24 GB | 8 | 32 | ~$0.98 | no — Edge only |
| `g5.xlarge` | A10G | Ampere | 24 GB | 4 | 16 | ~$1.00 | no — Edge only |
| `g6e.xlarge` | L40S | Ada | 44.7 GB | 4 | 32 | ~$1.86 | yes, **arch unsupported** |
| `g7.2xlarge` | RTX PRO 4500 | Blackwell | ~32 GB `?` | 8 `?` | `?` | ~$2.52 | borderline |
| **`g7e.2xlarge`** | RTX PRO 6000 | **Blackwell** | **96 GB** | 8 | 64 | **$5.719** eu-central-1 | yes, **supported** |
| `g7e.4xlarge` | RTX PRO 6000 | Blackwell | 96 GB | 16 | 128 | ~$4.00 | yes |
| `p5.4xlarge` | H100 | **Hopper** | 80 GB | 16 | 256 | ~$6.88 | yes, supported |

Multi-GPU instances (`p5.48xlarge`, `p6-b200`, `p6-b300`) are irrelevant here — a 4B
model spread across eight GPUs is waste.

### Recommendation

**`g7e.2xlarge` ($5.719/hr in eu-central-1) — CHOSEN.** See [`DECISIONS.md §4b`](DECISIONS.md).

Cosmos-Reason2 documents **Hopper and Blackwell** support. G7e is Blackwell, so the
architecture question disappears rather than being carried as a risk. It costs about
1.8x `g6e.xlarge` and returns double the VRAM (96 GB vs 44.7) and double the vCPU
(8 vs 4) — and 4 vCPU is thin for CPU-side video decoding alongside inference.
Against a total budget of roughly $25-40, that premium is about $15 to remove the
largest unknown in this document.

**`g6e.xlarge` (~$1.86/hr)** remains the cheaper bet if you would rather test the Ada
assumption early and keep the fallback in hand.

**`g6.xlarge` (~$0.80/hr)** is enough if only Cosmos3-Edge is ever served — but the
comparison models are what make any single number interpretable, so this saves money
only by removing the part that gives the result meaning.

### A two-instance option

Most GPU time goes on the primary model: the capability probe, the fps sweep, the
window/stride sweep are all Cosmos3-Edge. Only the final comparison needs 32 GB.

So a cheap `g6.xlarge` for the bulk, then a short `g7e.2xlarge` session for the
comparison, is genuinely cheaper. It costs two provisioning cycles and two weight
downloads. Worth it only if GPU budget is tight; otherwise one instance is simpler
and the simplicity is worth more than the difference.

### The serve command, confirmed on hardware

Flags carried over from NVIDIA's **Cosmos3-Nano** recipe needed one change, found
only by serving for real:

    --media-io-kwargs '{"video": {"num_frames": -1}}'      REMOVED

Two reasons. It broke — the JSON is double-quoted inside an already double-quoted
make variable, so the shell strips the inner quotes and vLLM rejects
`{video: {num_frames: -1}}`. And re-quoting it would be pointless: that flag
controls how vLLM decodes a **video file** it is handed, and this system never uses
that path. Frames are decoded here, timestamped, and sent as images.

Working flags:

    --max-model-len 32768 --allowed-local-media-path /data

Also noted, not yet acted on: vLLM warns that `--model` as an option is deprecated
in favour of a positional argument. Harmless while the version is pinned; it will
break on an upgrade.

### The vLLM version, and why the documented one fails

**`v0.29.0`, not the `v0.21.0`** that NVIDIA's Cosmos3-Nano recipe implies for
CUDA 13 drivers. 0.21.0 fails before reaching the GPU:

    The checkpoint you are trying to load has model type `cosmos3_edge`
    but Transformers does not recognize this architecture.

Its bundled Transformers predates the model. 0.29.0 resolves the architecture as
`Cosmos3EdgeForConditionalGeneration` and loads. **Eight minor versions between the
documented recipe and one that works** — and nothing in either the model card or
the recipe says so.

### Measured on hardware

`g7e.2xlarge`, NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 595.91.07,
CUDA 13.2. Figures that were previously estimates or entirely unpublished:

| | Measured | Previously |
|---|---|---|
| Total VRAM | **97,887 MiB** | 96 GB (spec) |
| Cosmos3-Edge checkpoint on disk | **7.19 GiB** | ~9 GB estimated |
| **Model weights in VRAM** | **4.67 GiB** | unpublished |
| Weight download | 70 s | unknown |
| Weight load | 8.55 s | unknown |
| Encoder cache budget | **24,300 tokens** | unknown |

The weights figure matters most: **4.67 GiB**, against Cosmos-Reason2-8B's
documented 32 GB minimum. A far smaller card would serve Edge alone — the 96 GB
instance was sized for the comparison models, not the primary.

The encoder cache budget is the first hard number bearing on **frames per call**,
which is what sets window length.

### Caveats

- **Prices are strongly region-dependent.** `g7e.2xlarge` is ~$3.36/hr in us-east-1
  but **$5.719/hr in eu-central-1** (console-verified, On-Demand Linux) — a ~70%
  premium. Every US figure quoted elsewhere understates the cost here. Always read
  the price in the launch console, in the region you will actually use.
- Region is dictated by the S3 bucket, not by price: the staging step only pays off
  if instance and bucket share a region.
- `g7.2xlarge` specs are `?` — RTX PRO 4500 VRAM and vCPU are not confirmed from a
  primary source. It sits right on the 32 GB line, so it is not recommended.
- **G7/G7e availability is region-limited.** Confirm your region offers it before
  planning around it.

---

## 3. Storage

### What has to persist

```
vLLM image        ~15 GB
finder image        2 GB
weights (all 4)    50 GB
OS + Docker        20 GB
                  ------
                   ~87 GB   must survive a stop
```

### What does not

Datasets — roughly 5 GB, and re-pullable from your S3 bucket in seconds once
staged. Nothing is lost by keeping them on ephemeral storage.

### The split

**EBS gp3 root: 150-200 GB.** Holds images and the weight cache, both of which
survive stop/start. Re-downloading 50 GB of weights every session would otherwise
cost more in GPU time than the disk costs in a month.

**Instance store for datasets, where available.** `g7e.2xlarge` ships **1900 GiB of
local NVMe**, and `g5`/`g6` families include NVMe too. It is *ephemeral* — wiped on
stop — but that is fine for clips that `make s3-pull` restores in seconds. It is
also considerably faster than EBS for repeated decoding passes.

To use it, mount the NVMe device and point `DATA_DIR` at it:

```bash
make s3-pull S3_BUCKET=<bucket> DATA_DIR=/mnt/nvme/data
make eval    DATASET=/mnt/nvme/data/synthetic/labels.json
```

**Never put the weight cache on instance store.** That is the one thing whose loss
actually costs money, and it is wiped on every stop.

**Stop, do not terminate.** Terminating destroys the EBS volume and the weight cache
with it.

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

Frankfurt (`eu-central-1`) on-demand Linux, console-verified for g7e.

| Activity | Time | `g7e.2xlarge` @ $5.719 |
|---|---|---|
| First serve: image pull + weights | ~30 min | ~$3 |
| Capability probe | ~1 hr | ~$6 |
| fps / window sweep | ~2 hr | ~$11 |
| Full model x clip matrix | ~3-5 hr | ~$17-29 |
| **Total with slack (~10-12 hr)** | | **~$60-70** |

Storage adds ~$0.08/GB-month, so a 200 GB gp3 root is about $16/month, accruing
even while the instance is stopped. S3 is negligible: 1.5 GB is about $0.04/month,
uploads are free, and same-region downloads to EC2 are free.

### Levers if that is too much

| Lever | Saving | Cost of it |
|---|---|---|
| **Stop the instance between sessions** | proportional | none — do this regardless |
| **Spot instances** | typically 60-70% | interruption at short notice; fine for re-runnable measurement, painful mid-sweep |
| **`g6e.xlarge` instead** | ~40%/hr | L40S is **Ada**, which NVIDIA does not list as supported — the risk this choice was made to avoid |
| **Drop the 8B comparison** | runs on a 24 GB card | removes the control that makes any single number interpretable |
| **Fewer clips** | proportional | the brief asks for 5-10; below that the eval stops being defensible |

Stopping between sessions is free and should be automatic. Spot is the next best
lever, and the measurement work is re-runnable by design, so an interruption costs
time rather than results.

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
not list these architectures as supported.

**This risk is avoidable.** An earlier version of this document claimed AWS had no
affordable single-GPU Hopper option; that was wrong. `p5.4xlarge` is a single H100
(Hopper, ~$6.88/hr), and the G7e family is Blackwell (`g7e.2xlarge`, 96 GB,
~$3.36/hr). Either satisfies NVIDIA's stated support matrix directly.

**So there are two routes:**

1. **Avoid it** — take `g7e.2xlarge`. Blackwell, supported, ~$1.50/hr more than
   `g6e.xlarge`.
2. **Accept it** — take `g6e.xlarge`, and serve `Cosmos-Reason2-8B` *early* in the
   first session rather than last. If Ada is a problem, it surfaces with the session
   still available to adapt, rather than after the budget is spent.
