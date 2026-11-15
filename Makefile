# One command per stage. `make paper` rebuilds every table and figure from the data.
PY ?= python
export PYTHONPATH := src:scripts

DATA      := data/midvdynattack/cropped
SPLITS    := data/splits_kfold_s0
CACHE     := data/cache_v2
CACHE_H264:= data/cache_h264
RES       := results/v3
PAPER     := results/paper

.PHONY: test data cache cache-h264 study paper traces cost testbed all

test:
	$(PY) -m pytest -q tests

data:            ## download the cropped MIDV-Holo / MIDV-DynAttack release and the 5-fold splits
	bash scripts/get_data.sh

cache:           ## still-image grid: JPEG/WebP x 3 scales x 8 qualities, MobileNetV3 embeddings
	$(PY) scripts/build_cache_v2.py --cropped_root $(DATA) --out $(CACHE)
	$(PY) scripts/build_cache_v2.py --cropped_root $(DATA) --out $(CACHE) --codecs webp \
	    --attacks no_holo plastified_lowreflect plastified_noholo laser plastified_led swap swap_three \
	              plain_holo leaf_holo double_sticker holo_star_world holo_completemask

cache-h264:      ## H.264 baseline (16 frames as one stream)
	$(PY) scripts/build_cache_h264.py --cropped_root $(DATA) --out $(CACHE_H264)

study:           ## all folds, policies, ablations, bootstrap statistics, unseen attacks
	$(PY) -m idpad.study_v3 --cache $(CACHE) --h264 $(CACHE_H264) --cropped_root $(DATA) \
	    --split_root $(SPLITS) --out $(RES) --unseen

paper:           ## Tables 4-10 and Figures 1-4 of the paper
	$(PY) scripts/fig_pipeline.py $(PAPER)/fig1_datapath.png
	$(PY) scripts/fig_system.py $(PAPER)/fig1_pipeline.png
	$(PY) scripts/make_paper_assets.py --res $(RES) --out $(PAPER)

traces:          ## Tables 11-12: published budget rules on measured cellular uplinks
	bash scripts/get_traces.sh
	$(PY) scripts/trace_comparison.py --boot 10000

cost:            ## Table 13: time per attempt on this machine
	$(PY) scripts/cost_benchmark.py --reps 50

testbed:         ## Table 14: Eq. (8) against a real TCP/TLS 1.3 stack (Linux, netem, user namespace)
	unshare -rn $(PY) scripts/testbed.py --reps 5 --out $(PAPER)/table12_testbed_raw.csv

all: test cache cache-h264 study paper traces cost
