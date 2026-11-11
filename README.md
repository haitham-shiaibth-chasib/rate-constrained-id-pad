# Frames or fidelity: rate-constrained hologram verification of identity documents

Code, protocol and results for the paper *Frames or Fidelity: Budgeting the Uplink for
Remote Hologram Verification of Identity Documents*.

A phone films an identity document while the user tilts it, and a server checks the
document's optically variable device (OVD, the hologram). On a weak uplink the phone can
only send a few kilobytes per attempt. This repository measures how those bytes should be
spent: more frames (K), higher quality, lower resolution, or another codec.

## What is in the study

| part | where |
| --- | --- |
| data loading, identity-leakage check | `src/idpad/data.py` |
| frozen MobileNetV3 embeddings, clip vector (mean, std, mean abs. frame difference) | `src/idpad/cnn.py` |
| still-image grid: JPEG/WebP x 3 scales x 8 qualities | `scripts/build_cache_v2.py` |
| H.264 baseline (16 frames as one stream) | `scripts/build_cache_h264.py` |
| policies, ablations, bootstrap CIs, paired tests, unseen attacks | `src/idpad/study_v3.py` |
| ISO/IEC 30107-3 metrics (BPCER20, APCER per species), EER | `src/idpad/metrics.py` |
| upload-time model (TCP + TLS 1.3, slow start) and link profiles | `src/idpad/network.py`, `configs/links.yaml` |
| hypotheses written before the five-fold run, and later additions | `docs/protocol_v2.md` |

## Reproduce the paper

```bash
pip install -r requirements-lock.txt     # exact versions used (Python 3.13, CPU only)
make test                                # 29 unit tests
make data                                # 7.1 GB from Zenodo + EPITA splits
make cache cache-h264                    # about 6 h on 2 CPU cores
make study                               # about 15 min
make paper                               # every table and figure
```

| paper item | produced by | file |
| --- | --- | --- |
| Figure 1 | `scripts/fig_pipeline.py` | `results/paper/fig1_pipeline.png` |
| Table 1 (source frames by K) | `make_paper_assets.py: table1` | `results/paper/table1.csv` |
| Figure 2 (EER vs budget per K) | `fig_k` | `fig2_k_vs_budget.png` |
| Table 2, Figure 3 (policies) | `policy_compact`, `fig_pol` | `table2c.csv` (with CIs: `table2.csv`), `fig3_policies.png` |
| Table 3 (paired comparisons) | `comparisons_compact` | `table3c.csv` (long form: `table3.csv`, `table4.csv`) |
| Figure 4 (unseen MIDV-DynAttack) | `fig_pol(unseen=True)` | `fig4_unseen.png`; numbers in `table5.csv`, per species `table5b.csv` |
| Table 4 (upload time) | `time_table` | `table6.csv` |
| Ablations quoted in Section 4.4 | `comparisons_wide` | `table4b.csv` |

`results/v3/` holds the outputs of `make study`: every configuration (`configs.csv`), the
validation-selected plans (`selection.csv`), test metrics per fold (`policies.csv`),
bootstrap intervals (`policy_ci.csv`), paired tests (`comparisons.csv`), unseen-attack
results and the per-clip scores (`scores.pkl`). All selections use validation data only;
the bootstrap uses seed 0 and 2,000 resamples.

## Data

* MIDV-Holo (Koliaskina et al., ICDAR 2023, CC BY-SA 2.5) and MIDV-DynAttack (Pouliquen
  et al., ICDAR 2025): cropped regions from Zenodo record
  [17079529](https://zenodo.org/records/17079529), file `cropped.tar.gz` (CC BY-SA 4.0).
* Five-fold identity-disjoint splits from
  [EPITAResearchLab/pouliquen.25.icdar](https://github.com/EPITAResearchLab/pouliquen.25.icdar).

The data are not redistributed here; `make data` downloads them.

## History

`idpad.pilot` (protocol v1, forensic-region coding with hand-crafted features) ended
NO-GO on fold 0 and is kept for transparency. `idpad.study_v2` is the first five-fold run;
`idpad.study_v3` replaces it with corrected metrics, uncertainty estimates and ablations.
The primary hypothesis H2 is not supported in v3.

## Licence

MIT (code). The datasets keep their own licences.
