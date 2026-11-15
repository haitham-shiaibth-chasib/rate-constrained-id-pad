"""Trace-driven version of the delivery model (Eq. 8) and the budget rules compared in Section 5.9.

A trace lists the times (ms) at which the bottleneck can deliver one 1500-byte packet, in the
Mahimahi format; the trace repeats when it ends. ``deliver`` follows the same rounds as
``Link.upload_seconds_tcp``: two handshake round trips, slow start from ten segments, a round
that lasts one RTT or until its last segment has crossed the link, whichever is later, and one
RTT after the last segment. On a constant-rate trace it returns the same times as Eq. 8.
"""
from __future__ import annotations

import math
import numpy as np

from .network import Link

GRID = (2000, 4000, 8000, 16000, 32000, 64000)


class Trace:
    def __init__(self, path: str):
        t = np.loadtxt(path).astype(np.int64)
        self.op, self.period = t, int(t[-1])
        n = int(math.ceil(self.period / 1000))
        # capacity of each one-second interval, kbit/s
        self.kbps = np.bincount((t - 1) // 1000, minlength=n)[:n] * 1500 * 8 / 1000.0

    def nth_after(self, t_ms: int, n: int) -> int:
        """Time (ms) of the n-th delivery opportunity strictly after ``t_ms``."""
        cyc, off = divmod(t_ms, self.period)
        j = int(np.searchsorted(self.op, off, side="right")) + n - 1
        c2, j2 = divmod(j, len(self.op))
        return (cyc + c2) * self.period + int(self.op[j2])


def deliver(tr: Trace, t0_ms: int, nbytes: int, rtt_ms: float, http_header: int = 600,
            mss: int = 1448, init_cwnd: int = 10, handshake_rtts: float = 2.0) -> float:
    """Seconds from the start of the connection to the reply, on trace ``tr``."""
    segs = math.ceil((nbytes + http_header) / mss)
    t, cwnd = t0_ms + handshake_rtts * rtt_ms, init_cwnd
    while segs > 0:
        n = min(cwnd, segs)
        tn = tr.nth_after(int(math.ceil(t)), n)
        segs -= n
        t = tn + rtt_ms if segs == 0 else max(t + rtt_ms, tn)
        cwnd *= 2
    return (t - t0_ms) / 1000.0


def harmonic(x) -> float:
    x = np.maximum(np.asarray(x, float), 1e-3)
    return len(x) / float(np.sum(1.0 / x))


def estimates(tr: Trace, s: int, w: int = 5):
    """Harmonic mean of the last ``w`` one-second capacities before second ``s``, and the
    RobustMPC version of it: divided by 1 + the largest relative error of the same predictor
    over the last ``w`` seconds (Yin et al., 2015), which uses 2w seconds of history. FESTIVE and
    RobustMPC use the last 20 and 5 chunk throughputs; one-second bins stand in for chunks here.
    A second without capacity is floored at 1e-3 kbit/s, so an outage inside the window drives
    every estimate, and hence every adaptive rule, to the smallest budget."""
    c = tr.kbps
    h = harmonic(c[s - w:s])
    err = max(abs(harmonic(c[k - w:k]) - max(c[k], 1e-3)) / max(c[k], 1e-3) for k in range(s - w, s))
    return h, h / (1.0 + err)


def _largest(grid, fits):
    ok = [b for b in grid if fits(b)]
    return max(ok) if ok else grid[0]


def choose(policy: str, h: float, hr: float, deadline: float, rtt_ms: float, grid=GRID, h20: float | None = None) -> int:
    """Budget (bytes) picked by each rule; the smallest budget when nothing fits. ``deadline`` is
    the time left for the network after encoding and server processing."""
    if policy.startswith("fixed"):
        return int(policy[5:]) * 1000
    if policy == "festive":       # rate-based rule, 0.85 x harmonic mean (Jiang et al., 2014)
        return _largest(grid, lambda b: b * 8 / 1000 / (0.85 * h) <= deadline)
    if policy == "festive20":     # FESTIVE with its original 20-sample window
        return _largest(grid, lambda b: b * 8 / 1000 / (0.85 * h20) <= deadline)
    if policy == "robustmpc":     # robust throughput estimate, size / throughput (Yin et al., 2015)
        return _largest(grid, lambda b: b * 8 / 1000 / hr <= deadline)
    if policy == "linear_setup":  # robust estimate plus three set-up round trips, no slow start
        return _largest(grid, lambda b: 3 * rtt_ms / 1000 + b * 8 / 1000 / hr <= deadline)
    if policy == "eq8":
        return Link("e", h, rtt_ms).budget_for_deadline(deadline, grid=grid) or grid[0]
    if policy == "eq8_robust":
        return Link("e", hr, rtt_ms).budget_for_deadline(deadline, grid=grid) or grid[0]
    raise ValueError(policy)
