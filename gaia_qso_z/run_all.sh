#!/usr/bin/env bash
# Full pipeline. Steps 1-2 need network access to data.sdss.org, gea.esac.esa.int and gaia.aip.de.
set -euo pipefail
cd "$(dirname "$0")"
for s in 01_crossmatch 02_fetch_xp 03_features 04_explore 05_regress 06_pz; do
  echo "=== $s ==="; python3 "$s.py" 2>&1 | tee "data/log_$s.txt"
done
