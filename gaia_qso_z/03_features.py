"""Step 3: build model features and calibrated spectra.

Features (one row per quasar):
  - c_bp_00..54, c_rp_00..54 : XP coefficients divided by the G-band flux (brightness removed;
                               the coefficients are linear in flux, so this is a pure shape vector)
  - e_bp_.., e_rp_..         : matching scaled 1-sigma errors
  - colours / magnitudes, DR16Q z, Gaia QSOC z
Calibrated spectra (GaiaXPy.calibrate) on 336-1020 nm are saved for the interpretable plots.
"""
import warnings

import numpy as np
import pandas as pd
from gaiaxpy import calibrate

from common import FEATURES, MATCH, N_COEF, SPECTRA, XP_RAW, XP_RANGE

SAMPLING = np.arange(XP_RANGE[0], XP_RANGE[1] + 0.1, 4.0)
CAL_BATCH = 2_000


def coef_matrix(xp, band, what="coefficients"):
    return np.vstack([np.asarray(v, float)[:N_COEF] for v in xp[f"{band}_{what}"]])


def main():
    m = pd.read_parquet(MATCH)
    xp = pd.read_pickle(XP_RAW)
    xp = xp[xp.source_id.isin(m.source_id)].reset_index(drop=True)
    m = m.set_index("source_id").loc[xp.source_id].reset_index()

    gflux = m.phot_g_mean_flux.to_numpy()
    feats = {}
    for band in ("bp", "rp"):
        C = coef_matrix(xp, band) / gflux[:, None]
        E = coef_matrix(xp, band, "coefficient_errors") / gflux[:, None]
        for k in range(N_COEF):
            feats[f"c_{band}_{k:02d}"] = C[:, k]
            feats[f"e_{band}_{k:02d}"] = E[:, k]
        feats[f"{band}_n_relevant_bases"] = xp[f"{band}_n_relevant_bases"].to_numpy()
        feats[f"{band}_n_measurements"] = xp[f"{band}_n_measurements"].to_numpy()
    f = pd.DataFrame(feats)
    keep = ["source_id", "qid", "z", "zwarning", "source_z", "bal_prob", "sep_arcsec",
            "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "phot_bp_rp_excess_factor",
            "redshift_qsoc", "redshift_qsoc_lower", "redshift_qsoc_upper", "flags_qsoc",
            "psfmag_u", "psfmag_g", "psfmag_r", "psfmag_i", "psfmag_z"]
    meta = m[[c for c in keep if c in m]].reset_index(drop=True)
    meta["bp_rp"] = meta.phot_bp_mean_mag - meta.phot_rp_mean_mag
    meta["bp_g"] = meta.phot_bp_mean_mag - meta.phot_g_mean_mag
    meta["g_rp"] = meta.phot_g_mean_mag - meta.phot_rp_mean_mag
    df = pd.concat([meta, f], axis=1)

    good = np.isfinite(df.z) & (df.z > 0) & np.isfinite(df[[c for c in df if c.startswith("c_")]]).all(axis=1)
    good &= np.isfinite(df.bp_rp)
    print(f"rows: {len(df):,}, usable: {good.sum():,}  (dropped {(~good).sum()} with bad z / NaN coeffs / no BP-RP)")
    df = df[good].reset_index(drop=True)
    xp = xp.set_index("source_id").loc[df.source_id].reset_index()
    df.to_parquet(FEATURES)

    # ---- calibrated, sampled spectra for interpretation ----
    cal_cols = ["source_id"] + [f"{b}_{c}" for b in ("bp", "rp") for c in (
        "n_parameters", "standard_deviation", "coefficients", "coefficient_errors", "coefficient_correlations",
        "basis_function_id", "n_relevant_bases", "relative_shrinking")]
    cal_cols = [c for c in cal_cols if c in xp]
    flux, ferr = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(xp), CAL_BATCH):
            spec, _ = calibrate(xp.iloc[i:i + CAL_BATCH][cal_cols].copy(), sampling=SAMPLING, save_file=False)
            flux.append(np.vstack(spec["flux"].to_numpy()))
            ferr.append(np.vstack(spec["flux_error"].to_numpy()))
            print(f"calibrated {min(i + CAL_BATCH, len(xp)):,}/{len(xp):,}", flush=True)
    np.savez_compressed(SPECTRA, source_id=df.source_id.to_numpy(), wl=SAMPLING,
                        flux=np.vstack(flux), flux_error=np.vstack(ferr), z=df.z.to_numpy())
    print("saved", FEATURES.name, SPECTRA.name)


if __name__ == "__main__":
    main()
