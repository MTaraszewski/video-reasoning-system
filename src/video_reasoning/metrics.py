"""Temporal metrics.

The brief asks for *"a temporal metric you can defend"*. Event-finding is
temporal grounding, so we report **temporal IoU** based measures, standard in the
moment-retrieval literature and the same family VANTAGE-Bench scores its Temporal
Localization task with.

Exact-boundary match is deliberately not used. Hand labels are fuzzy and so is the
model; a threshold on overlap is the honest bar, and reporting several thresholds
rather than one shows where accuracy actually falls off.

**The isolation rule.** Every metric is computed per `(video, description)` and
then aggregated. Pooling all predictions and all labels into flat lists and
matching on the description string would let a prediction from one clip satisfy a
label in another — and query strings repeat across clips *by design*, since the
same phrase is asked of several videos. That mistake inflates every number while
looking entirely reasonable, so the grouping is enforced in code rather than left
to a convention.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict

Interval = tuple[float, float]

IOU_THRESHOLDS = (0.3, 0.5, 0.7)

# Axes that are diagnostic, not representative. They are reported on their own
# and excluded from headline metrics.
#
#   positive_control  MEVA example clips with the activity label burned into the
#                     picture. A ceiling test: if the model cannot find an event
#                     whose name is written on the frame, it will not find it on
#                     clean footage. Scoring them alongside real clips would
#                     measure reading, not seeing.
CONTROL_AXES = {"positive_control"}


def t_iou(a: Interval, b: Interval) -> float:
    """Temporal intersection over union."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def relative_error(pred: Interval, truth: Interval) -> float:
    """Mean boundary error as a fraction of the true event's duration.

    This is NVIDIA's metric for temporal localisation — their recipe reports mean
    relative error with a success criterion of <30%. Included alongside tIoU so
    our results are comparable with published numbers rather than only internally
    consistent.
    """
    dur = max(1e-6, truth[1] - truth[0])
    return (abs(pred[0] - truth[0]) + abs(pred[1] - truth[1])) / (2 * dur)


def _key(item: dict) -> tuple[str, str]:
    """The isolation key. Both parts matter — see the module docstring."""
    return (item.get("video", ""), item.get("description", ""))


def _group(items: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for i in items:
        out[_key(i)].append(i)
    return out


def recall_at_1(preds: list[dict], truths: list[dict], thr: float) -> float:
    """Does the top-confidence prediction for this video+description overlap truth?

    Recall@1 asks whether the system's *best guess* is right, which is what a
    client actually experiences — they read the first result, not the tenth.
    """
    by_pred, by_truth = _group(preds), _group(truths)
    hits = total = 0
    for key, gts in by_truth.items():
        cands = sorted(by_pred.get(key, []),
                       key=lambda p: -p.get("confidence", 0.0))
        top = cands[:1]
        for g in gts:
            total += 1
            if top and t_iou((top[0]["start_s"], top[0]["end_s"]),
                             (g["start_s"], g["end_s"])) >= thr:
                hits += 1
    return round(hits / total, 4) if total else 0.0


def mean_best_iou(preds: list[dict], truths: list[dict]) -> float:
    """Mean, over true events, of the best overlap any prediction achieved."""
    by_pred, by_truth = _group(preds), _group(truths)
    vals: list[float] = []
    for key, gts in by_truth.items():
        cands = [(p["start_s"], p["end_s"]) for p in by_pred.get(key, [])]
        for g in gts:
            gt = (g["start_s"], g["end_s"])
            vals.append(max((t_iou(c, gt) for c in cands), default=0.0))
    return round(st.fmean(vals), 4) if vals else 0.0


def mean_relative_error(preds: list[dict], truths: list[dict]) -> float | None:
    """NVIDIA's metric, over matched pairs only.

    Unmatched truths are absent here on purpose: relative error is undefined
    without a prediction, and substituting a large number would silently blend a
    recall failure into a precision measure. Misses are reported by recall.
    """
    by_pred, by_truth = _group(preds), _group(truths)
    vals: list[float] = []
    for key, gts in by_truth.items():
        cands = [(p["start_s"], p["end_s"]) for p in by_pred.get(key, [])]
        if not cands:
            continue
        for g in gts:
            gt = (g["start_s"], g["end_s"])
            best = max(cands, key=lambda c: t_iou(c, gt))
            if t_iou(best, gt) > 0:
                vals.append(relative_error(best, gt))
    return round(st.fmean(vals), 4) if vals else None


def detection_pr(preds: list[dict], truths: list[dict], thr: float) -> tuple[float, float]:
    """Greedy one-to-one matching by descending confidence, within each group."""
    by_pred, by_truth = _group(preds), _group(truths)
    tp = 0
    for key in set(by_pred) | set(by_truth):
        gts = list(by_truth.get(key, []))
        matched: set[int] = set()
        for p in sorted(by_pred.get(key, []), key=lambda p: -p.get("confidence", 0.0)):
            best_j, best = -1, 0.0
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                i = t_iou((p["start_s"], p["end_s"]), (g["start_s"], g["end_s"]))
                if i > best:
                    best, best_j = i, j
            if best >= thr and best_j >= 0:
                tp += 1
                matched.add(best_j)
    precision = tp / len(preds) if preds else 0.0
    recall = tp / len(truths) if truths else 0.0
    return round(precision, 4), round(recall, 4)


def false_positive_rate(preds: list[dict], truths: list[dict]) -> float:
    """Fraction of predictions on video+description pairs that have NO true event.

    This is what negative clips measure, and why the eval set contains them:
    recall alone flatters a system that reports events everywhere.
    """
    truth_keys = {_key(t) for t in truths}
    if not preds:
        return 0.0
    spurious = sum(1 for p in preds if _key(p) not in truth_keys)
    return round(spurious / len(preds), 4)


def aggregate(preds: list[dict], truths: list[dict]) -> dict:
    """Every metric, computed with per-(video, description) isolation."""
    p50, r50 = detection_pr(preds, truths, 0.5)
    out: dict = {
        "n_predictions": len(preds),
        "n_truths": len(truths),
        "mean_tIoU": mean_best_iou(preds, truths),
        "precision@0.5": p50,
        "recall@0.5": r50,
        "false_positive_rate": false_positive_rate(preds, truths),
        "mean_relative_error": mean_relative_error(preds, truths),
    }
    for thr in IOU_THRESHOLDS:
        out[f"R@1_tIoU{thr}"] = recall_at_1(preds, truths, thr)
    return out


def by_axis(preds: list[dict], truths: list[dict], axes: dict[str, str]) -> dict:
    """Break results down by failure axis — the answer to 'where does it break?'

    A single aggregate number hides exactly what the brief asks for. Splitting by
    axis is what turns a score into a finding.
    """
    out: dict = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in truths:
        groups[axes.get(t.get("video", ""), "unknown")].append(t)
    for axis, gts in sorted(groups.items()):
        videos = {g["video"] for g in gts}
        ps = [p for p in preds if p.get("video") in videos]
        out[axis] = aggregate(ps, gts)
    return out
