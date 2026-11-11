"""ISO/IEC 30107-3 style metrics. Convention: label 1 = attack, score high = attack."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def _roc(labels: np.ndarray, scores: np.ndarray):
    """ROC points at every distinct threshold (same as sklearn's roc_curve with
    drop_intermediate=False, without its overhead; used inside bootstrap loops)."""
    o = np.argsort(-scores, kind="mergesort")
    s, y = scores[o], labels[o]
    idx = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = idx + 1 - tps
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]]


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal error rate: the point where APCER = BPCER, linearly interpolated on the ROC.

    Along the ROC, d = APCER - BPCER never increases, so the crossing lies between the
    last point with d > 0 and the first point with d <= 0.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr = _roc(labels, scores)  # fpr = BPCER, 1 - tpr = APCER
    d = (1 - tpr) - fpr
    j = int(np.argmax(d <= 0))
    if d[j] == 0 or j == 0:
        return float(fpr[j])
    a = d[j - 1] / (d[j - 1] - d[j])
    return float(fpr[j - 1] + a * (fpr[j] - fpr[j - 1]))


def bpcer_at_apcer(labels: np.ndarray, scores: np.ndarray, apcer: float = 0.05) -> float:
    """Lowest BPCER whose APCER <= ``apcer``. With apcer = 0.05 this is BPCER20 (ISO/IEC 30107-3).

    A small tolerance keeps operating points such as 4/80 = 0.05 that floating point
    would otherwise place just above the limit.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = (1 - tpr) <= apcer + 1e-9
    return float(fpr[ok].min()) if ok.any() else 1.0


def apcer_bpcer_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    labels, scores = np.asarray(labels), np.asarray(scores)
    atk, bona = scores[labels == 1], scores[labels == 0]
    apcer = float((atk < threshold).mean()) if len(atk) else float("nan")
    bpcer = float((bona >= threshold).mean()) if len(bona) else float("nan")
    return apcer, bpcer


def threshold_at_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Score threshold at the first ROC point where APCER <= BPCER (set on validation)."""
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    j = int(np.argmax((1 - tpr) - fpr <= 0))
    return float(min(thr[j], 1.0))


def max_apcer_per_species(labels, scores, species, threshold: float) -> tuple[float, dict]:
    """ISO/IEC 30107-3 reports the worst APCER over attack species (APCER_PAIS)."""
    labels, scores, species = np.asarray(labels), np.asarray(scores), np.asarray(species)
    per = {}
    for s in np.unique(species[labels == 1]):
        m = (labels == 1) & (species == s)
        per[str(s)] = float((scores[m] < threshold).mean())
    return (max(per.values()) if per else float("nan")), per


def bootstrap_ci(labels, scores, fn=eer, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI by resampling clips (the clip is the independent unit)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[i])) < 2:
            continue
        vals.append(fn(labels[i], scores[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
