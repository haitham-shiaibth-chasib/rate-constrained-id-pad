"""Identity-level bootstrap for the paper's intervals and paired tests.

Each test fold has 20 synthetic identities with about seven clips each. This script reuses
the statistics of idpad.study_v3 unchanged except for the resampling unit: it draws
identities within each fold and keeps every clip of a drawn identity. It writes
results/v3_identity/{policy_ci,comparisons}.csv and the Table 4 intervals (paper numbering).

Run from the repository root after `make study`:
    python scripts/identity_bootstrap.py --boot 10000
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "src")
try:
    import idpad.study_v3 as S
except ImportError:  # statistics only: the CNN module (torch) is not needed here
    import types
    _c = types.ModuleType("idpad.cnn"); _c.clip_embedding = None; sys.modules["idpad.cnn"] = _c
    import idpad.study_v3 as S
from idpad.data import _CLIP_RE


def identity(name):
    m = _CLIP_RE.match(name)
    return f"{m['dtype']}/{m['model']}_{m['ident']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/v3")
    ap.add_argument("--out", default="results/v3_identity")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pickle.load(open(res / "scores.pkl", "rb"))
    sel = pd.read_csv(res / "selection.csv")
    groups = {}
    for f, m in d["metas"].items():
        ids = np.array([identity(n) for n in m["names_test"]])
        groups[id(m["y_test"])] = [np.nonzero(ids == u)[0] for u in np.unique(ids)]

    def _boot_idx(y, rng):
        g = groups[id(y)]
        return np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))])

    S._boot_idx = _boot_idx
    cells, comps = S.statistics(sel, d["scores"], d["metas"], a.boot)
    cells.to_csv(out / "policy_ci.csv", index=False)
    comps.to_csv(out / "comparisons.csv", index=False)
    rows = []
    for k in (1, 2, 4, 8, 16):
        per = [(d["metas"][f]["y_test"], d["scores"][(f, "test", "clean", k, "spread", "ref", 1.0, -1)])
               for f in sorted(d["metas"])]
        lo, hi = S.boot_cell(per, a.boot)
        rows.append(dict(K=k, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(out / "table1_ci.csv", index=False)
    print(comps.round(4).to_string())


if __name__ == "__main__":
    main()
