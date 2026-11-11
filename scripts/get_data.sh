#!/usr/bin/env bash
# Download the public data used in the paper. About 7.1 GB.
set -euo pipefail
mkdir -p data && cd data
if [ ! -d midvdynattack/cropped ]; then
  curl -L -o cropped.tar.gz "https://zenodo.org/records/17079529/files/cropped.tar.gz?download=1"
  echo "8727c965d2a8588256cea303910eb1b7  cropped.tar.gz" | md5sum -c -
  mkdir -p midvdynattack && tar -xzf cropped.tar.gz -C midvdynattack && rm cropped.tar.gz
fi
if [ ! -d splits_kfold_s0 ]; then
  git clone --depth 1 https://github.com/EPITAResearchLab/pouliquen.25.icdar.git epita
  cp -r epita/data/splits_kfold_s0 . && rm -rf epita
fi
echo "data ready"
