"""Frame embeddings from an ImageNet-pretrained MobileNetV3 (phone-class backbone).

A clip is summarised by temporal statistics of its frame embeddings: mean, standard
deviation and mean absolute frame-to-frame change. The last two carry the temporal
behaviour of optically variable devices, which a single frame cannot show.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_MODEL = None
MODEL_NAME = "mobilenetv3_large_100.ra_in1k"
INPUT = 224
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def model():
    global _MODEL
    if _MODEL is None:
        import timm

        _MODEL = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval()
    return _MODEL


def _tensor(frames: list) -> torch.Tensor:
    arr = []
    for f in frames:
        im = f if isinstance(f, Image.Image) else Image.fromarray(f)
        im = im.convert("RGB").resize((INPUT, INPUT), Image.BILINEAR)
        arr.append(np.asarray(im, dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2)
    return (x - _MEAN) / _STD


@torch.no_grad()
def frame_embeddings(frames: list, batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(frames), batch):
        out.append(model()(_tensor(frames[i:i + batch])).numpy())
    return np.concatenate(out)


def clip_embedding(emb: np.ndarray) -> np.ndarray:
    """(K, d) frame embeddings -> (3d,) clip vector. Temporal parts are 0 when K = 1."""
    mean = emb.mean(0)
    if len(emb) < 2:
        z = np.zeros_like(mean)
        return np.concatenate([mean, z, z]).astype(np.float32)
    return np.concatenate([mean, emb.std(0), np.abs(np.diff(emb, axis=0)).mean(0)]).astype(np.float32)
