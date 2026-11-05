# Quasar redshifts from Gaia DR3 XP coefficients

Do the Gaia DR3 BP/RP (XP) Hermite coefficients carry quasar redshift information?
SDSS DR16Q spectroscopic redshifts are matched to Gaia DR3 sources with published XP
coefficients (36,402 quasars), and redshift is predicted from the coefficients with
cross-validated kNN, LightGBM and MLP models, compared with Gaia colours and Gaia's QSOC.

## Run

```bash
pip install -r requirements.txt
./run_all.sh            # or run 01_ ... 06_ in order
```

GaiaXPy 2.1.5 depends on `hdfs`/`docopt`, whose source builds can fail on Python 3.13; if so,
install `docopt-ng` and copy the pure-Python `hdfs` package from its sdist into site-packages.

Network hosts needed: `data.sdss.org` (DR16Q, 1.7 GB), `gea.esac.esa.int` (Gaia TAP),
`gaia.aip.de` (XP coefficients via TAP; ~8 min for all spectra). PyPI for installs.

## Scripts

| File | Step |
|---|---|
| `common.py` | paths, emission-line list, metrics, TAP upload helper (pyvo) |
| `01_crossmatch.py` | DR16Q → Gaia DR3 via DR16Q's DR2 ids and `gaiadr3.dr2_neighbourhood`; QSOC redshifts |
| `02_fetch_xp.py` | XP coefficients from the AIP mirror (default) or ESA DataLink (`--source esa`, slow) |
| `03_features.py` | flux-normalised coefficients; GaiaXPy-calibrated spectra |
| `04_explore.py` | redshift ladder, coefficient–z mutual information, PCA |
| `05_regress.py` | 5-fold CV regression: colours vs coefficients vs QSOC; shuffled-z control |
| `06_pz.py` | p(z) via MLP classifier over log(1+z) bins |

Outputs go to `data/` and `figures/`. `REPORT.pdf` summarises the results.
