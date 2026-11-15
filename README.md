# Frames or fidelity: rate-constrained hologram verification of identity documents

Code, protocol and results for the paper *Deadline-Aware Configuration of Capture and
Encoding for Remote Identity-Document Verification over Constrained Uplinks*.

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
| delivery-time model (TCP + TLS 1.3, slow start), its inverse for a deadline, and link profiles | `src/idpad/network.py`, `configs/links.yaml` |
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
| Tables 1 to 3 (related work, notation, settings) | no script (literature and settings) | – |
| Figure 1 (a: management loop, b: data path) | `scripts/fig_pipeline.py` (panel b), `scripts/fig_system.py` | `results/paper/fig1_datapath.png`, `results/paper/fig1_pipeline.png` |
| Table 4 (source frames by K) | `make_paper_assets.py: table1`; intervals by `scripts/identity_bootstrap.py` | `results/paper/table1.csv`, `results/v3_identity/table1_ci.csv` |
| Figure 2 (EER vs budget per K) | `fig_k` | `fig2_k_vs_budget.png` |
| Table 5, Figure 3 (policies) | `policy_compact`, `fig_pol` (error bars from `results/v3_identity/policy_ci.csv`) | `table2c.csv`, `fig3_policies.png` |
| Table 6 (paired comparisons) | `scripts/identity_bootstrap.py --boot 10000` | `results/v3_identity/comparisons.csv` |
| Figure 4 (unseen MIDV-DynAttack) | `fig_pol(unseen=True)` | `fig4_unseen.png`; numbers in `table5.csv`, per species `table5b.csv` |
| Table 7 (delivery time) | `time_table` | `table6.csv` |
| Table 8 (delivery regimes) | `scripts/deadline_tables.py` | `table5_regimes.csv` |
| Table 9 (deadline-driven configuration on GPRS) | `scripts/deadline_tables.py` (plan table: `src/idpad/plan_table.py`) | `table6_gprs.csv` |
| Table 10 (two-stage escalation, known attacks) | `scripts/escalation.py --boot 10000` (plan table as in Table 9) | `table7_escalation.csv`, `table7_escalation_tests.csv` |
| Tables 11 and 12 (published budget rules on measured cellular uplinks) | `bash scripts/get_traces.sh && python scripts/trace_comparison.py --boot 10000` (processing time from `table11_cost.json`) | `table8_late.csv`, `table8_good.csv`, `table8_tests.csv`, `table8_late_by_trace.csv`, `trace_attempts.csv.gz` |
| Table 13 (computational cost per attempt) | `python scripts/cost_benchmark.py --reps 50` | `table11_cost.json` |
| Table 14 (testbed check of Eq. 8 on real TCP + TLS 1.3) | `unshare -rn python3 scripts/testbed.py --reps 5` (Linux with netem; no root needed) | `table12_testbed_raw.csv`, `table12_testbed.csv` |
| Ablations quoted in Section 5.4 | `comparisons_wide` | `table4b.csv` |

Output file names predate the renumbering of the paper's tables (for example `table6_gprs.csv` is Table 9); the map above is authoritative. In `results/paper`, `table1.csv`, `table2c.csv` and `table3c.csv` carry clip-level intervals from `make study`; the intervals and stars printed in Tables 4 to 6 of the paper come from `results/v3_identity` (`table1_ci.csv`, `policy_ci.csv`, `comparisons.csv`).

Intervals and paired tests in the paper resample synthetic identities within each fold
(`scripts/identity_bootstrap.py`, 10,000 resamples, seed 0). `results/v3/policy_ci.csv` and
`results/v3/comparisons.csv` keep the clip-level bootstrap of `make study` for comparison;
the unseen-attack comparisons use the clip-level bootstrap because per-clip unseen scores
are not stored.

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

## Cellular uplink traces

Section 5.9 (Tables 11 and 12) replays three uplink traces (Verizon 1xEV-DO, T-Mobile UMTS, AT&T LTE) recorded by Winstein et al. (NSDI 2013) and distributed with Mahimahi (https://github.com/ravinet/mahimahi, GPL-3.0). They are not stored here; `scripts/get_traces.sh` downloads them from a fixed commit and checks their MD5 sums. `src/idpad/traces.py` holds the trace-driven delivery model and the budget rules; `tests/test_traces.py` checks that a constant-rate trace reproduces Eq. (8). The comparison runs in a few seconds and needs no GPU.
