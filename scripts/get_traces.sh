#!/usr/bin/env bash
# Download the three cellular uplink traces used in Section 5.9 (Mahimahi repository,
# traces recorded by Winstein et al., NSDI 2013). Each line of a trace is the time (ms) of
# one 1500-byte delivery opportunity.
set -euo pipefail
mkdir -p data/traces && cd data/traces
REV=f1346c38c415abfc98626dd82450f49a57033603
for f in Verizon-EVDO-driving.up TMobile-UMTS-driving.up ATT-LTE-driving.up; do
  [ -f "$f" ] || curl -sSL -o "$f" "https://raw.githubusercontent.com/ravinet/mahimahi/$REV/traces/$f"
done
md5sum -c - <<'SUM'
51184c683fb39221f8f34d8810ceecbf  Verizon-EVDO-driving.up
a6ab913410450d65820874e8b20ff8bd  TMobile-UMTS-driving.up
cb4dc26550a06dc98fe8ab9ea020ed44  ATT-LTE-driving.up
SUM
echo "traces ready"
