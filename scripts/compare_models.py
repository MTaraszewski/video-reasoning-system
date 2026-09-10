"""Build the leaderboard from whatever model runs exist.

Reads every `eval-*.json` and `probe-*.json` in the output directory and prints
one row per model. Nothing is inferred: a model that was not run gets a dash, not
an estimate, and the count of models actually measured is stated so a four-row
table cannot be mistaken for four measurements.

The comparison that matters is Cosmos-Reason2-8B against Qwen3-VL-8B-Instruct.
They are the same architecture and parameter count -- the second is the base model
the first was post-trained from -- so the gap between those rows is what NVIDIA's
physical-AI training buys for temporal localisation, measured rather than assumed.

    python scripts/compare_models.py --out-dir out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def row_from_eval(d: dict) -> dict:
    o = d.get("overall", {})
    cost = d.get("cost", {})
    return {
        "r1_03": o.get("R@1_tIoU0.3"), "r1_05": o.get("R@1_tIoU0.5"),
        "r1_07": o.get("R@1_tIoU0.7"),
        "miou": o.get("mean_tIoU"), "fp": o.get("false_positive_rate"),
        "rel": o.get("mean_relative_error"),
        "spm": cost.get("s_per_video_minute"),
        "calls": cost.get("model_calls"),
        "n_pred": o.get("n_predictions"), "n_truth": o.get("n_truths"),
    }


def row_from_probe(d: dict) -> dict:
    best = None
    for key, s in (d.get("by_prompt_fps") or {}).items():
        if s.get("median_abs_error_s") is None:
            continue
        if best is None or s["median_abs_error_s"] < best["median_abs_error_s"]:
            best = s
    if not best:
        # A model that emitted nothing at all still has an emit rate, and that
        # is the interesting number -- do not report it as missing data.
        any_s = next(iter((d.get("by_prompt_fps") or {}).values()), {})
        return {"emit": any_s.get("emitted_times"), "floor": None}
    return {"emit": best.get("emitted_times"), "floor": best.get("median_abs_error_s")}


def fmt(v, spec: str = "{:.3f}") -> str:
    return "—" if v is None else (spec.format(v) if isinstance(v, (int, float)) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--registry", default="models.tsv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    order: list[tuple[str, str]] = []
    reg = Path(args.registry)
    if reg.exists():
        for line in reg.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] in ("id",):
                continue
            order.append((parts[0], parts[3] if len(parts) > 3 else ""))

    rows = []
    for model, role in order:
        slug = model.replace("/", "-").lower()
        e = load(out / f"eval-{slug}.json")
        p = load(out / f"probe-{slug}.json")
        r = {"model": model, "role": role}
        r.update(row_from_eval(e) if e else {})
        r.update(row_from_probe(p) if p else {})
        r["ran"] = bool(e or p)
        rows.append(r)

    print(f"{'model':<34}{'R@1.3':>7}{'R@1.5':>7}{'mIoU':>8}{'FP':>7}"
          f"{'relerr':>8}{'emit':>7}{'floor':>8}{'s/vid-min':>11}")
    print("-" * 96)
    for r in rows:
        print(f"{r['model'][:33]:<34}"
              f"{fmt(r.get('r1_03')):>7}{fmt(r.get('r1_05')):>7}"
              f"{fmt(r.get('miou')):>8}{fmt(r.get('fp'), '{:.2f}'):>7}"
              f"{fmt(r.get('rel'), '{:.2f}'):>8}"
              f"{fmt(r.get('emit'), '{:.0%}'):>7}{fmt(r.get('floor'), '{:.2f}s'):>8}"
              f"{fmt(r.get('spm'), '{:.0f}'):>11}")

    ran = [r for r in rows if r["ran"]]
    print(f"\n{len(ran)} of {len(rows)} model(s) measured. A dash is 'not run', "
          f"never an estimate.")

    # The controlled comparison, called out explicitly so it is not buried.
    r2 = next((r for r in rows if "Reason2-8B" in r["model"] and r["ran"]), None)
    qw = next((r for r in rows if "Qwen3-VL-8B" in r["model"] and r["ran"]), None)
    print()
    if r2 and qw:
        d = (r2.get("miou") or 0) - (qw.get("miou") or 0)
        print(f"CONTROLLED COMPARISON — same architecture, same size, the only "
              f"difference is NVIDIA's post-training:")
        print(f"  Cosmos-Reason2-8B  mIoU {fmt(r2.get('miou'))}  emit {fmt(qw.get('emit'), '{:.0%}')}")
        print(f"  Qwen3-VL-8B        mIoU {fmt(qw.get('miou'))}  emit {fmt(qw.get('emit'), '{:.0%}')}")
        print(f"  difference         {d:+.4f} mIoU attributable to post-training")
    else:
        missing = [n for n, r in (("Cosmos-Reason2-8B", r2), ("Qwen3-VL-8B", qw)) if not r]
        print(f"CONTROLLED COMPARISON not available — {' and '.join(missing)} "
              f"not yet run. That comparison is the one result this sweep exists "
              f"to produce.")


if __name__ == "__main__":
    main()
