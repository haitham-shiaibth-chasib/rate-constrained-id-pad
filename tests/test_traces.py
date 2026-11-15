import numpy as np
from idpad.network import Link
from idpad.traces import Trace, deliver, GRID


def test_constant_trace_matches_eq6(tmp_path):
    for kbps, rtt in [(960, 100), (240, 300)]:
        gap = 1500 * 8 / kbps
        p = tmp_path / f"c{kbps}.tr"
        p.write_text("\n".join(str(int(np.ceil(i * gap))) for i in range(1, 100000)))
        tr, L = Trace(str(p)), Link("x", kbps, rtt)
        for b in GRID:
            assert abs(deliver(tr, 5000, b, rtt) - L.upload_seconds_tcp(b)) < 0.002
