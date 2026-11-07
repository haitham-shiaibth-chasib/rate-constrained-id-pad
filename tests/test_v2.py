import numpy as np
import pandas as pd

from idpad.study_v2 import decision, nested, transmit_lookup


def _entry():
    rng = np.random.default_rng(0)
    e = {"ref_emb": rng.normal(size=(16, 8)).astype(np.float16), "ref_bytes": np.full(16, 30000)}
    for c in ("jpeg", "webp"):
        for s in (1.0, 0.5):
            e[f"{c}_{s}_bytes"] = np.tile(np.array([200, 400, 800, 1200, 1800, 2500, 3500, 6000])[:, None], (1, 16))
            e[f"{c}_{s}_emb"] = rng.normal(size=(8, 16, 8)).astype(np.float16)
    return e


def test_nested():
    assert list(nested(1)) == [0] and list(nested(4)) == [0, 4, 8, 12]
    assert set(nested(4)) <= set(nested(8))


def test_lookup_budget():
    e = _entry()
    vec, nb = transmit_lookup(e, 4, "jpeg", 1.0, 5000)
    assert nb == 4800 and vec.shape == (24,)
    assert transmit_lookup(e, 16, "jpeg", 1.0, 2000) is None
    _, nb_ref = transmit_lookup(e, 2, None, None, None)
    assert nb_ref == 60000


def test_decision():
    rows = []
    for f in range(5):
        for b in (8000, 4000, 2000):
            rows += [dict(fold=f, budget=b, policy="photo", eer=0.30),
                     dict(fold=f, budget=b, policy="video16", eer=0.20 if b > 2000 else np.nan),
                     dict(fold=f, budget=b, policy="adaptive", eer=0.12)]
    d = decision(pd.DataFrame(rows))
    assert d["decision"] == "GO" and d["budgets_passing"] == 3
