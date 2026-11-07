import numpy as np
import pytest
from PIL import Image

from idpad import data as D
from idpad.codecs import available_codecs, encode_to_budget
from idpad.features import chroma_dynamics_map, clip_descriptor, dynamics_score, N_FRAME_STATS, N_DYNAMICS
from idpad.gate import evaluate_gate
from idpad.metrics import bpcer_at_apcer, eer
from idpad.network import Link
from idpad.transmission import Plan, prior_to_box, transmit


def _img(seed=0, size=(200, 140)):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, size[0] * size[1] * 3).reshape(size[1], size[0], 3)
    return Image.fromarray((base + rng.normal(0, 20, base.shape)).clip(0, 255).astype(np.uint8))


@pytest.mark.parametrize("codec", available_codecs())
@pytest.mark.parametrize("budget", [20000, 4000, 1500])
def test_encode_respects_budget(codec, budget):
    enc = encode_to_budget(_img(), codec, budget)
    assert enc is not None
    assert enc.nbytes <= budget
    assert enc.decode((200, 140)).size == (200, 140)


def test_encode_impossible_budget_returns_none():
    assert encode_to_budget(_img(), "jpeg", 50) is None


@pytest.mark.parametrize("strategy", ["uniform", "chroma", "roi"])
def test_transmit_total_bytes_within_budget(strategy):
    frames = [_img(i) for i in range(6)]
    sent = transmit(frames, Plan(strategy, "jpeg", 3), 9000, roi=(0.5, 0.4, 0.9, 0.9))
    assert sent.feasible and len(sent.frames) == 3
    assert 0 < sent.nbytes <= 9000
    assert all(f.size == frames[0].size for f in sent.frames)


def test_reference_condition_sends_original_frames():
    frames = [_img(i) for i in range(5)]
    sent = transmit(frames, Plan("uniform", "jpeg", 2), None)
    assert sent.nbytes == 0 and len(sent.frames) == 2


def test_dynamics_detects_colour_change():
    static = [_img(0)] * 4
    arrs = [np.asarray(_img(0)).copy() for _ in range(4)]
    for t, a in enumerate(arrs):
        a[40:100, 60:140, t % 3] = 255  # colour changes over time in one region
    dyn = [Image.fromarray(a) for a in arrs]
    assert dynamics_score(dyn) > dynamics_score(static) + 0.05
    m = chroma_dynamics_map(dyn)
    box = prior_to_box(m / m.max(), 0.2)
    assert box[0] < 0.75 and box[2] > 0.25  # box overlaps the changing region
    assert dynamics_score(dyn[:1]) == 0.0


def test_descriptor_length():
    assert clip_descriptor([_img(1), _img(2)]).shape == (2 * N_FRAME_STATS + N_DYNAMICS,)
    assert clip_descriptor([_img(1)]).shape == (2 * N_FRAME_STATS + N_DYNAMICS,)


def test_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert eer(y, np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])) == pytest.approx(0.0)
    assert eer(y, -np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])) == pytest.approx(1.0)
    assert bpcer_at_apcer(y, np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]), 0.05) == 0.0


def test_link_budget_roundtrip():
    link = Link("sat", 512, 650, 0.01)
    b = link.budget_for_deadline(5.0)
    assert b > 0 and link.upload_seconds(b) == pytest.approx(5.0, rel=1e-3)


def test_identity_key_and_leakage():
    a = D.Clip("ID/id01_03_01", None, 0, "bona_fide", ())
    b = D.Clip("ID/id01_03_02", None, 1, "copy_without_holo", ())
    c = D.Clip("ID/id01_04_01", None, 0, "bona_fide", ())
    assert a.identity_key == b.identity_key == "ID/01_03"
    assert D.check_leakage([a], [c]) == {}
    assert D.check_leakage([a], [b]) == {"0-1": {"ID/01_03"}}


def test_gate_logic():
    import pandas as pd
    rows = []
    for f in range(5):
        for split in ("val", "test"):
            for bud in (-1, 4000, 2000):
                for plan, strat, e in (("uniform-jpeg-K1", "uniform", 0.05 if bud == -1 else 0.30),
                                       ("roi-jpeg-K3-r0.60", "roi", 0.05 if bud == -1 else 0.10)):
                    rows.append(dict(fold=f, split=split, plan=plan, strategy=strat, budget=bud,
                                     detector="descriptor_lr", eer=e, feasible=1.0, mean_bytes=bud))
    v = evaluate_gate(pd.DataFrame(rows), dict(baseline_plan="uniform-jpeg-K1", min_degradation_eer=0.05,
                                               low_budgets=[4000, 2000], min_relative_gain=0.2,
                                               min_budgets_with_gain=2, min_folds_improved=4,
                                               detector="descriptor_lr"))
    assert v["decision"] == "GO"
