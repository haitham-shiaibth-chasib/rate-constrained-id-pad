"""Indexing of the MIDV-Holo / MIDV-DynAttack cropped ROIs and the public k-fold splits.

Expected layout (as distributed in Zenodo record 17079529, `cropped.tar.gz`):

    <cropped_root>/origins/ID/id01_01_01/img_0001.jpg ...
    <cropped_root>/fraud/<attack_type>/ID/id01_01_01/img_0001.jpg ...

Split files follow the public protocol of Pouliquen et al. (ICDAR 2024/2025):

    <split_root>/k{fold}/simple/{train,val,test}.txt          -> bona fide clips
    <split_root>/k{fold}/fraud/<attack_type>/{train,test}.txt -> attack clips

Clip names look like ``ID/id01_03_02``: document type, template (model) 01,
synthetic identity 03, clip 02. Splitting by identity is what keeps frames of the
same physical document out of both train and test; ``check_leakage`` verifies it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Attack folders that belong to the original MIDV-Holo release.
MIDV_HOLO_ATTACKS = ("copy_without_holo", "pseudo_holo_copy", "photo_holo_copy", "photo_replacement")
# Attack folders introduced by MIDV-DynAttack (test-only by design of that dataset).
DYNATTACK_ATTACKS = (
    "no_holo", "plastified_lowreflect", "plastified_noholo", "laser", "plastified_led",
    "swap", "swap_three", "plain_holo", "leaf_holo", "double_sticker", "holo_star_world",
    "holo_completemask",
)

_CLIP_RE = re.compile(r"^(?P<dtype>[A-Za-z]+)/(?P<prefix>[a-z]+)(?P<model>\d+)_(?P<ident>\d+)_(?P<clip>\d+)$")


@dataclass(frozen=True)
class Clip:
    """One video clip of one physical document (bona fide or attack)."""

    name: str            # e.g. "ID/id01_03_02"
    directory: Path
    label: int           # 0 = bona fide, 1 = attack (ISO/IEC 30107-3 convention: attack is positive)
    attack_type: str     # "bona_fide" or the attack folder name
    frames: tuple[Path, ...] = field(repr=False)

    @property
    def identity_key(self) -> str:
        """Document type + template + synthetic identity (the unit that must not leak)."""
        m = _CLIP_RE.match(self.name)
        if not m:
            return self.name
        return f"{m['dtype']}/{m['model']}_{m['ident']}"

    @property
    def template_key(self) -> str:
        m = _CLIP_RE.match(self.name)
        return f"{m['dtype']}/{m['model']}" if m else self.name


def _normalise_name(line: str) -> str:
    line = line.strip().strip("/")
    if line.endswith(".lst") or line.endswith(".txt"):
        line = line.rsplit("/", 1)[0]
    return line


def read_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [_normalise_name(l) for l in path.read_text().splitlines() if l.strip()]


def list_frames(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(p for p in directory.glob("*.jpg") if p.is_file()))


def _make_clips(root: Path, names: Iterable[str], label: int, attack_type: str, min_frames: int) -> list[Clip]:
    clips = []
    for n in names:
        d = root / n
        if not d.is_dir():
            continue
        frames = list_frames(d)
        if len(frames) < min_frames:
            continue
        clips.append(Clip(n, d, label, attack_type, frames))
    return clips


def load_fold(
    cropped_root: str | Path,
    split_root: str | Path,
    fold: int,
    part: str,
    attacks: Iterable[str] = MIDV_HOLO_ATTACKS,
    min_frames: int = 10,
) -> list[Clip]:
    """Return bona fide + attack clips of one fold partition.

    ``part`` is "train", "val" or "test". Attack splits only exist as train/test in the
    public protocol, so for "val" the attack clips are taken from the attack *train*
    list restricted to identities that appear in the bona fide val list.
    For DynAttack folders (no own split files) the bona fide test list is reused,
    exactly as in the public configuration files.
    """
    cropped_root, split_root = Path(cropped_root), Path(split_root)
    kdir = split_root / f"k{fold}"
    bona_names = read_split(kdir / "simple" / f"{part}.txt")
    clips = _make_clips(cropped_root / "origins", bona_names, 0, "bona_fide", min_frames)
    val_ids = {Clip(n, Path(), 0, "", ()).identity_key for n in bona_names} if part == "val" else None

    for atk in attacks:
        atk_root = cropped_root / "fraud" / atk
        if not atk_root.is_dir():
            continue
        own = kdir / "fraud" / atk
        if own.is_dir():
            src = "train" if part == "val" else part
            names = read_split(own / f"{src}.txt")
        else:
            names = bona_names  # DynAttack convention
        atk_clips = _make_clips(atk_root, names, 1, atk, min_frames)
        if val_ids is not None:
            atk_clips = [c for c in atk_clips if c.identity_key in val_ids]
        elif part == "train":
            # keep attack train clips away from identities reserved for validation
            val_names = read_split(kdir / "simple" / "val.txt")
            reserved = {Clip(n, Path(), 0, "", ()).identity_key for n in val_names}
            atk_clips = [c for c in atk_clips if c.identity_key not in reserved]
        clips.extend(atk_clips)
    return clips


def check_leakage(*partitions: list[Clip]) -> dict[str, set[str]]:
    """Identity keys shared between any two partitions (must be empty)."""
    keys = [{c.identity_key for c in p} for p in partitions]
    overlaps: dict[str, set[str]] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = keys[i] & keys[j]
            if inter:
                overlaps[f"{i}-{j}"] = inter
    return overlaps


def sample_frames(clip: Clip, n: int) -> list[Path]:
    """``n`` frames spread evenly over the clip (deterministic)."""
    f = clip.frames
    if n >= len(f):
        return list(f)
    step = len(f) / n
    return [f[int(i * step)] for i in range(n)]


def describe(clips: list[Clip]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in clips:
        out[c.attack_type] = out.get(c.attack_type, 0) + 1
    return out
