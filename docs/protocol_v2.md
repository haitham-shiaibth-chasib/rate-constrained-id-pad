# Study protocol v2 (fixed before running the v2 experiments)

## Why a second protocol

The pilot (v1, `configs/pilot.yaml`) ended NO-GO on fold 0:

* a single frame carries almost no evidence of an optically variable device, so the
  v1 baseline (one JPEG frame) was at chance level even without compression, and the
  "problem exists" condition could not be met by construction;
* the forensic-ROI layer learned from bona fide clips covered ~70 % of the ID card and
  did not beat uniform coding at equal bytes;
* the hand-crafted detector reached only 23 % EER on uncompressed clips.

v1 is kept in the repository and will be reported as a negative result.

## Research question

With a fixed uplink payload per verification attempt, how should a phone split the
bytes between **number of frames (K)**, **per-frame quality / resolution** and
**codec**, to minimise hologram presentation-attack detection error?

## Detector

ImageNet-pretrained MobileNetV3-Large embeddings per frame; clip vector = mean,
standard deviation and mean absolute frame-to-frame change of the embeddings; logistic
regression (C = 0.1, balanced classes), trained on uncompressed training clips for each K.
Reference EER on fold 0 (uncompressed): 30.0 % at K = 1, 8.8 % at K = 32.

## Transmission model

16 frames sampled evenly per clip; K in {1, 2, 4, 8, 16} uses nested subsets. One
quality setting is used for all K frames of an attempt (as a phone encoder would).
Grid: codec in {JPEG 4:2:0, WebP}, scale in {1.0, 0.5}, eight quality levels. For a
budget B the plan picks, per clip, the highest quality whose total size of the K frames
is <= B. Budgets B in {64, 32, 16, 8, 4, 2} kB plus the uncompressed reference.

## Policies compared at each budget

* `photo`: K = 1, best codec/scale chosen on validation (single ID photo, current practice)
* `video16`: K = 16, best codec/scale chosen on validation (send the whole burst)
* `adaptive`: K, codec and scale chosen on validation (proposed)

## Hypotheses and decision rule

* H1: the EER-optimal K grows with the budget.
* H2 (primary): `adaptive` has at least 20 % lower test EER than **both** `photo` and
  `video16` at three or more of the six budgets, in at least four of five folds.
* H3: WebP gives lower EER than JPEG at budgets <= 8 kB.

GO if H2 holds. Secondary analyses: unseen MIDV-DynAttack attacks (test only),
upload time per link profile in `configs/links.yaml`.

Note fixed in advance: a policy whose plan cannot fit the budget for every clip is
"not deployable" at that budget; for H2 an undeployable baseline counts as beaten.
A plan is only scored when it is feasible for all clips of the split.
