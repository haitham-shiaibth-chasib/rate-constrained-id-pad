"""Pre-registered go / no-go decision for the pilot.

(a) Problem exists: the conventional baseline (one JPEG 4:2:0 frame) loses at least
    ``min_degradation_eer`` EER points between the reference and the low budgets.
(b) Contribution exists: at equal bytes, the best forensic-aware plan (chroma or roi),
    chosen on validation, beats the best uniform plan, also chosen on validation, by at
    least ``min_relative_gain`` (relative test EER) in ``min_folds_improved`` folds, at
    ``min_budgets_with_gain`` or more budgets.
Both selections use validation EER only; test EER is read once for the selected plans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _select(val: pd.DataFrame, test: pd.DataFrame, fold: int, budget: int, strategies: set[str]):
    v = val[(val.fold == fold) & (val.budget == budget) & (val.strategy.isin(strategies)) & (val.feasible == 1.0)]
    v = v.dropna(subset=["eer"])
    if v.empty:
        return None, np.nan
    best = v.sort_values(["eer", "mean_bytes"]).iloc[0]["plan"]
    t = test[(test.fold == fold) & (test.budget == budget) & (test.plan == best)]
    return best, (float(t.eer.iloc[0]) if len(t) else np.nan)


def evaluate_gate(df: pd.DataFrame, g: dict) -> dict:
    df = df[df.detector == g.get("detector", "descriptor_lr")]
    val, test = df[df.split == "val"], df[df.split == "test"]
    folds = sorted(test.fold.unique())

    # (a) degradation of the baseline
    base = test[test.plan == g["baseline_plan"]]
    deg = []
    for f in folds:
        b = base[base.fold == f]
        ref = b[b.budget == -1].eer.mean()
        low = b[b.budget.isin(g["low_budgets"])].eer.mean()
        deg.append(low - ref)
    deg_mean = float(np.nanmean(deg)) if deg else float("nan")
    cond_a = bool(deg_mean >= g["min_degradation_eer"])

    # (b) forensic-aware vs uniform at equal budget, both selected on validation
    per_budget = {}
    for bud in sorted(b for b in test.budget.unique() if b != -1):
        gains, details = [], []
        for f in folds:
            pu, eu = _select(val, test, f, bud, {"uniform"})
            pf, ef = _select(val, test, f, bud, {"chroma", "roi"})
            gain = (eu - ef) / eu if eu and not np.isnan(eu) and not np.isnan(ef) and eu > 0 else np.nan
            gains.append(gain)
            details.append(dict(fold=int(f), uniform=pu, uniform_eer=eu, forensic=pf, forensic_eer=ef, rel_gain=gain))
        gains = np.array(gains, dtype=float)
        improved = int(np.nansum(gains >= g["min_relative_gain"]))
        per_budget[int(bud)] = dict(mean_rel_gain=float(np.nanmean(gains)) if np.isfinite(gains).any() else None,
                                    folds_improved=improved,
                                    passes=bool(improved >= g["min_folds_improved"]),
                                    folds=details)
    n_pass = sum(v["passes"] for v in per_budget.values())
    cond_b = bool(n_pass >= g["min_budgets_with_gain"])
    return dict(
        condition_a_problem_exists=cond_a, baseline_eer_degradation=deg_mean, per_fold_degradation=deg,
        condition_b_contribution=cond_b, budgets_passing=n_pass, per_budget=per_budget,
        decision="GO" if (cond_a and cond_b) else "NO-GO",
    )
