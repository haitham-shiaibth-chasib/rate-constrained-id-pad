"""The offline step of Sec. 3: the plan table p*(B) used by the deadline rule.

For each fold and budget the table takes, among the candidate policies whose plan fits every
clip, the plan with the lowest validation EER. Two rules make the table well defined:

* ties (equal validation EER within ``tol``) keep the policy used at the next smaller budget,
  then prefer fewer frames, then the order of ``CANDIDATES``; the table therefore changes
  only when validation evidence asks for it;
* the table is cumulative: a plan selected at budget B' also fits every B > B', so if the best
  plan at B is worse on validation than the best plan at a smaller budget, the smaller plan is
  kept. min_p EER_val(p, B) is then non-increasing in B, which Eq. (2) needs.
"""
from __future__ import annotations

import pandas as pd

CANDIDATES = ["photo", "adaptive", "video16", "video16_h264", "video16_h264_ca"]
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]


def rule_table(sel: pd.DataFrame, candidates=CANDIDATES, budgets=BUDGETS, tol: float = 1e-9):
    """Return {(fold, budget): selection row} with the extra keys 'source_budget' and 'tied'."""
    out = {}
    for f in sorted(sel.fold.unique()):
        prev, best = None, None
        for b in budgets:
            x = sel[(sel.fold == f) & (sel.budget == b) & sel.policy.isin(candidates) & (sel.feasible == True)]
            if x.empty:
                continue
            m = x.val_eer.min()
            ties = x[x.val_eer <= m + tol].copy()
            ties["order"] = ties.policy.map({p: i for i, p in enumerate(candidates)})
            ties["is_prev"] = ties.policy == prev
            r = ties.sort_values(["is_prev", "k", "order"], ascending=[False, True, True]).iloc[0].copy()
            r["tied"] = len(ties) > 1
            if best is not None and best.val_eer < m - tol:      # cumulative table
                r = best.copy(); r["tied"] = False
            else:
                r["source_budget"] = b
                best = r
            out[(f, b)] = r
            prev = r.policy
    return out
