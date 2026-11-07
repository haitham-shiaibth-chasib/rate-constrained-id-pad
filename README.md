# Rate-constrained presentation attack detection for remote ID-document verification

Remote onboarding to e-government services asks citizens to film their identity document
with a phone. Where the uplink is a satellite or weak 3G link, the capture must be
compressed hard before it is sent, and the forensic cues that reveal a printed or
screen-displayed copy (hologram colour shifts, fine print texture) are the first details
lost. This repository measures that loss and tests whether a transmission plan that
protects forensic content keeps detection error low at the same number of bytes.

## What is measured

For every verification attempt the client has a payload budget in bytes. A **plan**
decides how the budget is spent:

| strategy | description |
| --- | --- |
| `uniform` | K frames, budget split evenly, standard 4:2:0 chroma |
| `chroma`  | same, but 4:4:4 chroma (holograms are a colour phenomenon) |
| `roi`     | two layers: a cheap base layer plus a high-quality layer of the forensic region, learned from training bona fide clips only |

Codecs: JPEG, WebP, AVIF. Every encoded payload is measured, so all comparisons are at
equal bytes. `idpad.network` converts payloads into upload time for configurable links
(`configs/links.yaml`).

Detectors: a training-free chroma-dynamics score and a logistic regression on a
43-dimensional clip descriptor. Metrics follow ISO/IEC 30107-3 (APCER, BPCER, EER,
BPCER at APCER = 5 %), with bootstrap confidence intervals over clips.

## Data

* MIDV-Holo (Koliaskina et al., ICDAR 2023) and MIDV-DynAttack (Pouliquen et al.,
  ICDAR 2025): cropped regions of interest from Zenodo record
  [17079529](https://zenodo.org/records/17079529), file `cropped.tar.gz` (CC BY-SA).
* Public 5-fold splits from
  [EPITAResearchLab/pouliquen.25.icdar](https://github.com/EPITAResearchLab/pouliquen.25.icdar)
  (`data/splits_kfold_s0`). Clips are split by synthetic identity; `idpad.data.check_leakage`
  stops the run if any identity appears in two partitions.

Neither the data nor the split files are redistributed here; the notebook downloads them.

## Quick start

```bash
pip install -e . && pip install -r requirements.txt
pytest                                   # unit tests
python scripts/make_synthetic.py --out data/synthetic
python -m idpad.pilot --config configs/synthetic_smoke.yaml   # code-path check only
```

The synthetic data exist only to test the code path; numbers obtained on them have no
scientific meaning.

For the real pilot use `notebooks/01_pilot_colab.ipynb`, or locally:

```bash
python scripts/dataset_report.py --cropped_root data/midvdynattack/cropped
python -m idpad.pilot --config configs/pilot.yaml
python scripts/plot_pilot.py
```

## Pre-registered decision rule

`configs/pilot.yaml` fixes the go / no-go thresholds before any result is seen:

1. the conventional baseline (one JPEG 4:2:0 frame) must lose at least 5 EER points at
   the low budgets, otherwise there is no problem to solve;
2. the best forensic-aware plan must beat the best uniform plan, both selected on
   validation, by at least 20 % relative test EER in at least 4 of 5 folds, at two or
   more budgets.

## Layout

```
src/idpad/   data.py  codecs.py  transmission.py  features.py  detectors.py
             metrics.py  network.py  pilot.py  gate.py
configs/     pilot.yaml  links.yaml  synthetic_smoke.yaml
scripts/     dataset_report.py  plot_pilot.py  make_synthetic.py
notebooks/   01_pilot_colab.ipynb
tests/       test_core.py
```

## License

Code: MIT. Data remain under their original licenses.
