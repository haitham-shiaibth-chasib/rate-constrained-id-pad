"""Uplink model that turns a payload size into end-to-end verification time.

The profiles are *assumptions*, listed in ``configs/links.yaml`` so a reviewer can
change them. Time = connection set-up (``setup_rtts`` x RTT) + payload / effective
goodput, where goodput accounts for packet loss with a simple TCP-like penalty
(Mathis et al., 1997) capped by the link rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Link:
    name: str
    uplink_kbps: float
    rtt_ms: float
    loss: float = 0.0
    setup_rtts: float = 3.0   # TCP + TLS 1.3 handshake + request
    mss_bytes: int = 1400

    def goodput_bps(self) -> float:
        rate = self.uplink_kbps * 1000.0
        if self.loss <= 0:
            return rate
        mathis = (self.mss_bytes * 8 / (self.rtt_ms / 1000.0)) * (1.22 / math.sqrt(self.loss))
        return min(rate, mathis)

    def upload_seconds(self, nbytes: int) -> float:
        return self.setup_rtts * self.rtt_ms / 1000.0 + nbytes * 8 / self.goodput_bps()

    def budget_for_deadline(self, deadline_s: float) -> int:
        """Largest payload that arrives within ``deadline_s`` (0 if impossible)."""
        t = deadline_s - self.setup_rtts * self.rtt_ms / 1000.0
        return max(0, int(t * self.goodput_bps() / 8))


DEFAULT_LINKS = (
    Link("geo_satellite_vsat", uplink_kbps=512, rtt_ms=650, loss=0.01),
    Link("3g_weak", uplink_kbps=256, rtt_ms=200, loss=0.02),
    Link("4g_typical", uplink_kbps=5000, rtt_ms=60, loss=0.005),
    Link("fiber_office", uplink_kbps=20000, rtt_ms=20, loss=0.0),
)


def load_links(path: str | None) -> tuple[Link, ...]:
    if path is None:
        return DEFAULT_LINKS
    import yaml

    with open(path) as f:
        items = yaml.safe_load(f)["links"]
    return tuple(Link(**it) for it in items)
