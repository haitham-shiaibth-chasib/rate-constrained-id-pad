import numpy as np

from idpad.network import Link
from idpad.study_v3 import boot_diff, frame_idx, holm


def test_frame_layouts():
    assert frame_idx(4, "spread").tolist() == [0, 4, 8, 12]
    assert frame_idx(4, "window").tolist() == [0, 1, 2, 3]
    assert set(frame_idx(2, "spread")) <= set(frame_idx(8, "spread"))


def test_holm_monotone_and_capped():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and abs(adj[2] - 0.06) < 1e-12 and abs(adj[1] - 0.06) < 1e-12
    assert max(holm([0.5, 0.9])) <= 1.0


def test_boot_diff_identical_scores_is_zero():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(30), np.ones(30)]
    s = rng.random(60)
    d, lo, hi, p = boot_diff([(y, s, s)] * 2, B=50)
    assert d == 0 and lo == 0 and hi == 0 and p == 1.0


def test_tcp_time_grows_with_payload_and_rtt():
    a, b = Link("a", 1000, 100), Link("b", 1000, 600)
    assert a.upload_seconds_tcp(64000) > a.upload_seconds_tcp(2000)
    assert b.upload_seconds_tcp(2000) > a.upload_seconds_tcp(2000)
    assert abs(a.upload_seconds_tcp(0) - 3 * 0.1 - (1448 + 52) * 8 / 1e6) < 1e-9
