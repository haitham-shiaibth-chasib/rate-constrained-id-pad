"""ISO/IEC 30107-3 style metrics. Convention: label 1 = attack, score high = attack."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)  # fpr = BPCER, 1 - tpr = APCER
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    # linear interpolation between neighbours for a less jumpy estimate
    if i + 1 < len(fpr):
        d0, d1 = fnr[i] - fpr[i], fnr[i + 1] - fpr[i + 1]
        if d0 != d1 and np.sign(d0) != np.sign(d1):
            a = d0 / (d0 - d1)
            return float(fpr[i] + a * (fpr[i + 1] - fpr[i]))
    return float((fpr[i] + fnr[i]) / 2)


def bpcer_at_apcer(labels: np.ndarray, scores: np.ndarray, apcer: float = 0.05) -> float:
    """Lowest BPCER whose APCER <= ``apcer`` (BPCER-at-APCER operating point)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = (1 - tpr) <= apcer
    return float(fpr[ok].min()) if ok.any() else 1.0


def apcer_bpcer_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    labels, scores = np.asarray(labels), np.asarray(scores)
    atk, bona = scores[labels == 1], scores[labels == 0]
    apcer = float((atk < threshold).mean()) if len(atk) else float("nan")
    bpcer = float((bona >= threshold).mean()) if len(bona) else float("nan")
    return apcer, bpcer


def threshold_at_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(labels, scores)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float(thr[i])


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
