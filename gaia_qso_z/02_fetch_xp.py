"""Step 2: fetch Gaia DR3 XP continuous coefficients (BP+RP, 55 each) for matched quasars.

Default source: the AIP Gaia mirror (gaia.aip.de), whose TAP service exposes
gaiadr3.xp_continuous_mean_spectrum, so an uploaded-id join fetches 2,000 spectra in seconds.
Fallback (--source esa): ESA's TAP does not expose that table, so source_id batches are POSTed to
the ESA DataLink service (50 ids per anonymous request, ~5 s per source: very slow).
Output: data/xp_continuous.pkl (archive column names, array columns as numpy arrays;
        this is the input format GaiaXPy expects).
"""
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from astropy.table import Table

from common import AIP_TAP, DATA, MATCH, XP_RAW, gaia_upload_query

BATCH = 1_000            # sources per saved file
PER_REQUEST = 50         # DataLink limit for anonymous users
THREADS = 4
DATALINK = "https://gea.esac.esa.int/data-server/data"
BATCH_DIR = DATA / "xp_batches"
BATCH_DIR.mkdir(exist_ok=True)
AIP_BATCH = 2_000
AIP_QUERY = """
SELECT x.* FROM TAP_UPLOAD.ids AS u
JOIN gaiadr3.xp_continuous_mean_spectrum AS x ON x.source_id = u.source_id
"""



def fetch_request(ids, retries=6):
    payload = {"RETRIEVAL_TYPE": "XP_CONTINUOUS", "DATA_STRUCTURE": "DATAMODEL_GAIA", "RELEASE": "Gaia DR3",
               "FORMAT": "votable", "ID": ",".join(map(str, ids))}
    for attempt in range(retries):
        try:
            r = requests.post(DATALINK, data=payload, timeout=(30, 600))
            r.raise_for_status()
            blob = r.content
            if blob[:2] == b"PK":   # zipped multi-file answer
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    tabs = [Table.read(io.BytesIO(z.read(n)), format="votable") for n in z.namelist()]
                from astropy.table import vstack
                return vstack(tabs)
            return Table.read(io.BytesIO(blob), format="votable")
        except Exception as e:
            wait = 2 ** (attempt + 2)
            print(f"  attempt {attempt + 1} failed ({str(e)[:150]}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("DataLink batch failed repeatedly")
ARRAY_COLS = [f"{b}_{c}" for b in ("bp", "rp")
              for c in ("coefficients", "coefficient_errors", "coefficient_correlations")]


def to_numpy_arrays(df):
    for c in ARRAY_COLS:
        if c in df:
            df[c] = [np.asarray(np.ma.filled(v, np.nan) if np.ma.isMaskedArray(v) else v, dtype=float)
                     for v in df[c]]
    return df


def fetch_aip(ids):
    n = int(np.ceil(len(ids) / AIP_BATCH))
    for i in range(n):
        out = BATCH_DIR / f"aip_{i:03d}.pkl"
        if out.exists():
            continue
        t0 = time.time()
        tab = gaia_upload_query(AIP_QUERY, Table({"source_id": ids[i * AIP_BATCH:(i + 1) * AIP_BATCH]}), "ids",
                                retries=6, poll=2, url=AIP_TAP)
        df = pd.DataFrame({c: list(tab[c]) if tab[c].ndim > 1 or tab[c].dtype == object else np.asarray(tab[c])
                           for c in tab.colnames})
        to_numpy_arrays(df).to_pickle(out)
        print(f"AIP batch {i + 1}/{n}: {len(df)} spectra ({time.time() - t0:.0f}s)", flush=True)
    return sorted(BATCH_DIR.glob("aip_*.pkl"))


def main():
    m = pd.read_parquet(MATCH)
    ids = m.loc[m.has_xp_continuous, "source_id"].astype(np.int64).to_numpy()
    # random order: every completed batch file is a random subsample, so the analysis can start
    # on a partial download (the archive serves ~5 s of CPU per source to anonymous users)
    ids = np.random.default_rng(0).permutation(np.sort(ids))
    print(f"Sources with XP: {len(ids):,}")
    n_batches = int(np.ceil(len(ids) / BATCH))
    if "esa" not in sys.argv[1:] or "--source" not in sys.argv:
        files = fetch_aip(ids)
        xp = pd.concat([pd.read_pickle(p) for p in files], ignore_index=True).drop_duplicates("source_id")
        print(f"XP spectra retrieved: {len(xp):,}  (of {len(ids):,} with XP)")
        print("bp/rp coefficient vector lengths:", {len(v) for v in xp.bp_coefficients}, {len(v) for v in xp.rp_coefficients})
        xp.to_pickle(XP_RAW)
        return
    assemble_only = "--assemble" in sys.argv
    for i in range(n_batches):
        if assemble_only:
            break
        out = BATCH_DIR / f"xp_{i:03d}.pkl"
        if out.exists():
            continue
        t0 = time.time()
        chunk = ids[i * BATCH:(i + 1) * BATCH]
        with ThreadPoolExecutor(THREADS) as ex:
            parts = list(ex.map(fetch_request, [chunk[j:j + PER_REQUEST] for j in range(0, len(chunk), PER_REQUEST)]))
        from astropy.table import vstack
        tab = vstack([t for t in parts if len(t)])
        df = pd.DataFrame({c: list(tab[c]) if tab[c].ndim > 1 or tab[c].dtype == object else np.asarray(tab[c])
                           for c in tab.colnames})
        to_numpy_arrays(df).to_pickle(out)
        print(f"batch {i + 1}/{n_batches}: {len(df)} spectra ({time.time() - t0:.0f}s)", flush=True)

    xp = pd.concat([pd.read_pickle(p) for p in sorted(BATCH_DIR.glob("xp_*.pkl"))], ignore_index=True)
    xp = xp.drop_duplicates("source_id")
    print(f"XP spectra retrieved: {len(xp):,}  (of {len(ids):,} with XP)")
    print("bp/rp coefficient vector lengths:", {len(v) for v in xp.bp_coefficients}, {len(v) for v in xp.rp_coefficients})
    xp.to_pickle(XP_RAW)


if __name__ == "__main__":
    main()
