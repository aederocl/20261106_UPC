"""Step 1: SDSS DR16Q x Gaia DR3 match.

DR16Q already carries its Gaia DR2 counterpart (GAIA_DESIGNATION, matched by the SDSS team).
DR2 ids are uploaded to the Gaia archive and mapped to DR3 through gaiadr3.dr2_neighbourhood,
joined to gaia_source. This id-based join is ~100x faster than a positional join, which the
archive aborts after 60 s for anonymous users. Per quasar we keep the DR3 neighbour closest
to the SDSS position, requiring < 1 arcsec.

A positional 1" cross-match of the first ~180k DR16Q rows (data/xmatch_batches/, from the earlier
positional run) is used to measure how complete the id route is for sources with XP spectra.

Outputs:
  data/dr16q_slim.parquet           slim DR16Q (one row per quasar, qid = row index)
  data/dr16q_gaia_xp_match.parquet  Gaia DR3 counterpart per quasar, has_xp_continuous, QSOC z
"""
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from common import DATA, DR16Q_FITS, DR16Q_SLIM, DR16Q_URL, MATCH, gaia_upload_query

BATCH = 20_000
G_DR2_MAX = 20.5          # XP spectra exist down to G ~ 20; skip fainter DR2 counterparts
RADIUS_ARCSEC = 1.0
ID_DIR = DATA / "idmatch_batches"
ID_DIR.mkdir(exist_ok=True)
POS_DIR = DATA / "xmatch_batches"

WANTED = ["SDSS_NAME", "RA", "DEC", "Z", "SOURCE_Z", "ZWARNING", "Z_PIPE", "BAL_PROB", "IS_QSO_FINAL",
          "GAIA_G_MAG"]

QUERY = """
SELECT u.qid, n.dr2_source_id, n.dr3_source_id AS source_id, n.angular_distance AS dr2_dr3_dist_mas,
       DISTANCE(POINT('ICRS', u.ra, u.dec), POINT('ICRS', g.ra, g.dec)) * 3600 AS sep_arcsec,
       g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
       g.phot_g_mean_flux, g.phot_bp_mean_flux, g.phot_rp_mean_flux,
       g.phot_bp_rp_excess_factor, g.has_xp_continuous, g.classprob_dsc_combmod_quasar,
       g.parallax, g.parallax_over_error, g.pm
FROM tap_upload.q AS u
JOIN gaiadr3.dr2_neighbourhood AS n ON n.dr2_source_id = u.dr2_source_id
JOIN gaiadr3.gaia_source AS g ON g.source_id = n.dr3_source_id
"""
QSOC_QUERY = """
SELECT q.source_id, q.redshift_qsoc, q.redshift_qsoc_lower, q.redshift_qsoc_upper, q.zscore_qsoc, q.flags_qsoc
FROM tap_upload.ids AS u JOIN gaiadr3.qso_candidates AS q ON q.source_id = u.source_id
"""


def download_dr16q():
    if DR16Q_FITS.exists():
        return
    print(f"Downloading {DR16Q_URL} ...", flush=True)
    tmp = DR16Q_FITS.with_suffix(".part")
    with urllib.request.urlopen(DR16Q_URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(DR16Q_FITS)


def native(v):
    return v.byteswap().view(v.dtype.newbyteorder()) if v.dtype.byteorder == ">" else v


def slim_dr16q():
    if DR16Q_SLIM.exists():
        df = pd.read_parquet(DR16Q_SLIM)
        if "dr2_source_id" in df:
            return df
    with fits.open(DR16Q_FITS, memmap=True) as h:
        d = h[1].data
        names = set(d.columns.names)
        cols = {c.lower(): native(np.asarray(d[c])) for c in WANTED if c in names}
        if "PSFMAG" in names:  # ugriz PSF magnitudes
            psf = native(np.asarray(d["PSFMAG"]))
            for i, b in enumerate("ugriz"):
                cols[f"psfmag_{b}"] = psf[:, i]
        des = np.char.strip(np.asarray(d["GAIA_DESIGNATION"]).astype(str))
    df = pd.DataFrame(cols).rename(columns={"gaia_g_mag": "g_dr2"})
    df["sdss_name"] = df["sdss_name"].astype(str).str.strip()
    df["dr2_source_id"] = np.array([int(s.split()[-1]) if s.startswith("Gaia") else -1 for s in des], dtype=np.int64)
    df.insert(0, "qid", np.arange(len(df), dtype=np.int64))
    df.to_parquet(DR16Q_SLIM)
    return df


def run_batches(q):
    up = q[(q.dr2_source_id > 0) & (q.g_dr2 < G_DR2_MAX)][["qid", "ra", "dec", "dr2_source_id"]]
    print(f"uploading {len(up):,} DR2 ids (G_DR2 < {G_DR2_MAX})")
    for start in range(0, len(up), BATCH):
        out = ID_DIR / f"ids_{start:07d}.parquet"
        if out.exists():
            continue
        t0 = time.time()
        res = gaia_upload_query(QUERY, Table.from_pandas(up.iloc[start:start + BATCH]), "q", retries=6)
        res.to_pandas().to_parquet(out)
        print(f"  {min(start + BATCH, len(up)):>7,}/{len(up):,}: {len(res)} rows ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat([pd.read_parquet(p) for p in sorted(ID_DIR.glob("ids_*.parquet"))], ignore_index=True)


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def validate(m):
    """Compare with the positional cross-match where it exists (first ~180k DR16Q rows)."""
    files = sorted(POS_DIR.glob("batch_q*.parquet"))
    if not files:
        return
    pos = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    covered = set()
    for p in files:                        # file names: batch_q<start>_<stop>.parquet
        a, b = (int(x) for x in p.stem.removeprefix("batch_q").split("_"))
        covered.update(range(a, b))
    pos = pos[as_bool(pos.has_xp_continuous)].sort_values("sep_arcsec").drop_duplicates("qid")
    idm = m[m.qid.isin(covered) & m.has_xp_continuous]
    both = pos.merge(idm[["qid", "source_id"]], on="qid", how="left", suffixes=("", "_id"))
    rec = both.source_id_id.notna().mean()
    same = (both.source_id == both.source_id_id)[both.source_id_id.notna()].mean()
    extra = (~idm.qid.isin(pos.qid)).sum()
    print(f"\nValidation on {len(covered):,} DR16Q rows that also have a positional match run:")
    print(f"  positional XP matches: {len(pos):,}; recovered by id route: {rec:.1%} (same DR3 source: {same:.1%});"
          f" id-route XP matches not in positional set: {extra}")
    lost = both[both.source_id_id.isna()]
    if len(lost):
        print("  G of missed sources, percentiles 10/50/90:", np.percentile(lost.phot_g_mean_mag, [10, 50, 90]).round(2))


def main():
    download_dr16q()
    q = slim_dr16q()
    print(f"DR16Q quasars: {len(q):,}; with a DR2 counterpart: {(q.dr2_source_id > 0).sum():,}")
    m = run_batches(q)
    m["has_xp_continuous"] = as_bool(m["has_xp_continuous"])
    m = m[m.sep_arcsec < RADIUS_ARCSEC]
    m = m.sort_values("sep_arcsec").drop_duplicates("qid", keep="first")
    m = m.sort_values("sep_arcsec").drop_duplicates("source_id", keep="first")
    m = m.merge(q.drop(columns=["dr2_source_id"]), on="qid", how="left")

    ids = m.loc[m.has_xp_continuous, "source_id"].astype(np.int64).to_numpy()
    qs = pd.concat([gaia_upload_query(QSOC_QUERY, Table({"source_id": ids[i:i + BATCH]}), "ids", retries=6).to_pandas()
                    for i in range(0, len(ids), BATCH)], ignore_index=True)
    m = m.merge(qs, on="source_id", how="left")
    m.to_parquet(MATCH)

    xp = m[m.has_xp_continuous]
    print("\n=== Cross-match summary ===")
    print(f"DR16Q quasars                 {len(q):>9,}")
    print(f"with a Gaia DR3 source <1\"    {len(m):>9,}  ({len(m) / len(q):.1%})  [G_DR2 < {G_DR2_MAX}]")
    print(f"... of which have XP coeffs   {len(xp):>9,}  ({len(xp) / len(q):.2%} of DR16Q)")
    print(f"... with Gaia QSOC redshift   {xp.redshift_qsoc.notna().sum():>9,}")
    print(f"median separation (XP set)    {xp.sep_arcsec.median():.3f}\"")
    print("XP-set G mag percentiles 5/50/95:", np.percentile(xp.phot_g_mean_mag.dropna(), [5, 50, 95]).round(2))
    print("XP-set z percentiles 5/50/95:   ", np.percentile(xp.z, [5, 50, 95]).round(2))
    validate(m)


if __name__ == "__main__":
    main()
