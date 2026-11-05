"""Shared paths, constants and helpers for the Gaia XP x SDSS quasar redshift study."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GQZ_DATA", ROOT / "data"))
FIG = Path(os.environ.get("GQZ_FIG", ROOT / "figures"))
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DR16Q_URL = "https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits"
DR16Q_FITS = DATA / "DR16Q_v4.fits"
DR16Q_SLIM = DATA / "dr16q_slim.parquet"
MATCH = DATA / "dr16q_gaia_xp_match.parquet"
XP_RAW = DATA / "xp_continuous.pkl"          # list/array columns -> pickle keeps dtypes intact
FEATURES = DATA / "features.parquet"
SPECTRA = DATA / "calibrated_spectra.npz"
PRED = DATA / "cv_predictions.parquet"

SEED = 42
N_COEF = 55

# Broad emission lines in the rest frame (nm)
LINES = {
    "Lyα": 121.6,
    "CIV": 154.9,
    "CIII]": 190.9,
    "MgII": 279.8,
    "[OII]": 372.7,
    "Hβ": 486.1,
    "[OIII]": 500.7,
    "Hα": 656.3,
}
XP_RANGE = (336.0, 1020.0)  # nm, GaiaXPy calibrated range


def metrics(z_true, z_pred):
    """Standard photo-z metrics on dz = (z_pred - z_true) / (1 + z_true)."""
    z_true = np.asarray(z_true, float)
    z_pred = np.asarray(z_pred, float)
    dz = (z_pred - z_true) / (1.0 + z_true)
    med = np.median(dz)
    return {
        "N": int(dz.size),
        "bias_median": float(med),
        "sigma_NMAD": float(1.4826 * np.median(np.abs(dz - med))),
        "outlier_0.1": float(np.mean(np.abs(dz) > 0.1)),
        "outlier_0.2": float(np.mean(np.abs(dz) > 0.2)),
        "frac_|dz|<0.01": float(np.mean(np.abs(z_pred - z_true) < 0.01 * (1 + z_true))),
    }


def confusion_ratios():
    """(1+z_pred)/(1+z_true) ratios produced by mistaking line i for line j."""
    names = list(LINES)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = LINES[b] / LINES[a]
            # observed λ = λ_a(1+z_true) = λ_b(1+z_pred)  =>  (1+z_pred)/(1+z_true) = λ_a/λ_b
            out.append((f"{a} read as {b}", 1 / r))
            out.append((f"{b} read as {a}", r))
    return out


GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
AIP_TAP = "https://gaia.aip.de/tap"   # AIP mirror: also exposes gaiadr3.xp_continuous_mean_spectrum


def gaia_upload_query(query, upload, name, retries=4, poll=5, max_wait=3 * 3600, url=GAIA_TAP):
    """Run an async ADQL query on a Gaia TAP service (ESA by default) with an uploaded astropy Table.

    Uses pyvo (requests honours the HTTPS proxy; astroquery's Gaia client does not).
    Retries on network errors and archive-side aborts.
    """
    import time

    import pyvo

    tap = pyvo.dal.TAPService(url)
    last = None
    for attempt in range(retries):
        job = None
        try:
            job = tap.submit_job(query, uploads={name: upload})
            job.run()
            t0 = time.time()
            while True:
                time.sleep(poll)
                phase = job.phase
                if phase in ("COMPLETED", "ERROR", "ABORTED"):
                    break
                if time.time() - t0 > max_wait:
                    raise TimeoutError("job did not finish")
            if phase != "COMPLETED":
                try:
                    job.raise_if_error()
                except Exception as err:
                    raise RuntimeError(f"job {phase}: {err}") from None
                raise RuntimeError(f"job {phase}")
            res = job.fetch_result().to_table()
            try:
                job.delete()
            except Exception:
                pass
            return res
        except Exception as e:
            last = e
            wait = 2 ** (attempt + 2)
            print(f"  attempt {attempt + 1} failed ({str(e)[:160]}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"query failed after {retries} attempts: {last}")
