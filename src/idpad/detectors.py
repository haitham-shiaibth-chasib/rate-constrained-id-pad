"""Two lightweight detectors. Both output an *attack* score (higher = more likely attack).

* ``DynamicsDetector``  - training-free; minus the chroma-dynamics score.
* ``DescriptorDetector`` - logistic regression on ``clip_descriptor`` (25 + 13 features);
  small enough to run on a phone, which is the deployment target of the study.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import clip_descriptor, dynamics_score


class DynamicsDetector:
    name = "dynamics"

    def fit(self, X_frames: list[list], y: np.ndarray) -> "DynamicsDetector":
        return self

    def score(self, X_frames: list[list]) -> np.ndarray:
        return np.array([-dynamics_score(f) for f in X_frames])


class DescriptorDetector:
    name = "descriptor_lr"

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed),
        )

    @staticmethod
    def describe(X_frames: list[list]) -> np.ndarray:
        return np.stack([clip_descriptor(f) for f in X_frames])

    def fit(self, X_frames: list[list], y: np.ndarray, D: np.ndarray | None = None) -> "DescriptorDetector":
        D = self.describe(X_frames) if D is None else D
        self.model.fit(D, y)
        return self

    def score(self, X_frames: list[list], D: np.ndarray | None = None) -> np.ndarray:
        D = self.describe(X_frames) if D is None else D
        return self.model.predict_proba(D)[:, 1]
