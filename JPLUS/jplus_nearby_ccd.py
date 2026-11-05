#!/usr/bin/env python3
"""
J-PLUS DR4 colour-colour diagrams of nearby stars (d < 500 pc, parallax error < 10%).

Selection
---------
  Gaia DR3 (jplus.xmatch_gaia_dr3): plx > 2 mas  (d = 1000/plx < 500 pc)
                                    e_plx < 0.1 * plx
  J-PLUS (jplus.MagABDualObj):      FLAGS < 4, MASK_FLAGS == 0, NORM_WMAP_VAL > 0.8
                                    in every filter used by the plots
  Magnitudes: MAG_APER_COR_3_0 (3" aperture, aperture-corrected)

Usage
-----
  pip install pyvo astropy matplotlib numpy
  python jplus_nearby_ccd.py            # query CEFCA TAP, save FITS, plot
  python jplus_nearby_ccd.py --replot   # plot again from the saved FITS
"""
import argparse

import numpy as np

TAP_URL = "https://archive.cefca.es/catalogues/vo/tap/jplus-dr4"

PLX_MIN_MAS = 2.0          # 500 pc
PLX_REL_ERR_MAX = 0.10     # 10 %
FLAGS_MAX = 4              # keep FLAGS < 4
WMAP_MIN = 0.8             # keep NORM_WMAP_VAL > 0.8
MAG_MAX = 30.0             # drop undetected values (99)

# Array order of the J-PLUS vector columns = jplus.Filter.filter_id (1..12)
FILTERS = ["rSDSS", "gSDSS", "iSDSS", "zSDSS", "uJAVA", "J0378",
           "J0395", "J0410", "J0430", "J0515", "J0660", "J0861"]

# Colour-colour diagrams: ((x_blue, x_red), (y_blue, y_red))
DIAGRAMS = [
    (("rSDSS", "iSDSS"), ("gSDSS", "rSDSS")),   # stellar locus; WDs off to the blue
    (("rSDSS", "iSDSS"), ("J0660", "rSDSS")),   # H-alpha: active M dwarfs fall below the locus
]

OUT_FITS = "jplus_dr4_nearby_raw.fits"
OUT_PNG = "jplus_nearby_ccd.png"
OUT_CLEAN = "jplus_dr4_nearby_clean.fits.gz"

WAVELENGTH = {"uJAVA": 3485, "J0378": 3785, "J0395": 3950, "J0410": 4100,
              "J0430": 4300, "gSDSS": 4750, "J0515": 5150, "rSDSS": 6250,
              "J0660": 6600, "iSDSS": 7725, "J0861": 8610, "zSDSS": 9150}

QUERY = f"""
SELECT m.tile_id, m.number, m.alpha_j2000, m.delta_j2000,
       m.flags, m.mask_flags, m.norm_wmap_val,
       m.mag_aper_cor_3_0, m.mag_err_aper_cor_3_0,
       x.source, x.angdist, x.plx, x.e_plx, x.ruwe, x.gmag, x.bp_rp
FROM jplus.MagABDualObj AS m
JOIN jplus.xmatch_gaia_dr3 AS x ON x.tile_id = m.tile_id AND x.number = m.number
WHERE x.plx > {PLX_MIN_MAS} AND x.e_plx < {PLX_REL_ERR_MAX} * x.plx"""


def fetch(n_chunks=8, n_parallel=4):
    """The CEFCA TAP returns at most 1,000,000 rows per job, so split by tile_id."""
    import pyvo
    from astropy.table import vstack
    from concurrent.futures import ThreadPoolExecutor
    svc = pyvo.dal.TAPService(TAP_URL)
    r = svc.search("SELECT MIN(tile_id) AS lo, MAX(tile_id) AS hi FROM jplus.TileImage").to_table()
    edges = np.linspace(int(r["lo"][0]), int(r["hi"][0]) + 1, n_chunks + 1).astype(int)

    def run(i):
        q = QUERY + f" AND m.tile_id >= {edges[i]} AND m.tile_id < {edges[i + 1]}"
        job = svc.submit_job(q, maxrec=10_000_000)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=7200)
        job.raise_if_error()
        t = job.fetch_result().to_table()
        job.delete()
        if len(t) >= 1_000_000:
            raise RuntimeError(f"chunk {i} hit the 1M-row cap; increase n_chunks")
        print(f"[tap] chunk {i + 1}/{n_chunks}: {len(t)} rows", flush=True)
        return t

    with ThreadPoolExecutor(n_parallel) as ex:
        t = vstack(list(ex.map(run, range(n_chunks))))
    t.write(OUT_FITS, overwrite=True)
    print(f"[tap] {len(t)} rows -> {OUT_FITS}")
    return t


def used_filters():
    return sorted({f for d in DIAGRAMS for pair in d for f in pair}, key=FILTERS.index)


def clean(t):
    mag = np.ma.filled(np.ma.asarray(t["mag_aper_cor_3_0"], float), np.nan)
    flags = np.asarray(t["flags"])
    mflags = np.asarray(t["mask_flags"])
    wmap = np.asarray(t["norm_wmap_val"], float)

    ok = np.ones(len(t), bool)
    for f in used_filters():
        i = FILTERS.index(f)
        ok &= ((flags[:, i] < FLAGS_MAX) & (mflags[:, i] == 0) & (wmap[:, i] > WMAP_MIN)
               & np.isfinite(mag[:, i]) & (mag[:, i] < MAG_MAX))
    print(f"[cuts] {len(t)} Gaia-matched rows -> {ok.sum()} pass J-PLUS cuts in {used_filters()}")
    t = t[ok]

    # one row per Gaia source (keep closest match)
    order = np.lexsort((np.asarray(t["angdist"], float), np.asarray(t["source"])))
    _, first = np.unique(np.asarray(t["source"])[order], return_index=True)
    n0 = len(t)
    t = t[np.sort(order[first])]
    print(f"[dedup] {n0} -> {len(t)} unique Gaia sources")
    t["mag"] = np.ma.filled(np.ma.asarray(t["mag_aper_cor_3_0"], float), np.nan)
    t["dist_pc"] = 1000.0 / np.asarray(t["plx"], float)
    return t


def save_clean(t, out=OUT_CLEAN):
    """Compact catalogue: one row per star, one column per filter, float32.

    A magnitude is NaN where that filter fails FLAGS/MASK_FLAGS/NORM_WMAP_VAL,
    so any colour built from these columns is already quality-filtered.
    """
    from astropy.table import Table
    flags = np.asarray(t["flags"])
    mflags = np.asarray(t["mask_flags"])
    wmap = np.asarray(t["norm_wmap_val"], float)
    mag = np.array(t["mag"], float)
    err = np.ma.filled(np.ma.asarray(t["mag_err_aper_cor_3_0"], float), np.nan)
    bad = ((flags >= FLAGS_MAX) | (mflags != 0) | (wmap <= WMAP_MIN)
           | ~np.isfinite(mag) | (mag >= MAG_MAX))
    mag[bad] = np.nan
    err[bad] = np.nan

    c = Table()
    c["gaia_source_id"] = np.asarray(t["source"], np.int64)
    c["ra"] = np.asarray(t["alpha_j2000"], np.float64)
    c["dec"] = np.asarray(t["delta_j2000"], np.float64)
    c["tile_id"] = np.asarray(t["tile_id"], np.int32)
    c["number"] = np.asarray(t["number"], np.int32)
    for col, src in [("plx", "plx"), ("e_plx", "e_plx"), ("dist_pc", "dist_pc"),
                     ("ruwe", "ruwe"), ("gaia_g", "gmag"), ("gaia_bp_rp", "bp_rp")]:
        c[col] = np.ma.filled(np.ma.asarray(t[src], float), np.nan).astype(np.float32)
    for k in sorted(range(len(FILTERS)), key=lambda k: WAVELENGTH[FILTERS[k]]):
        c[f"mag_{FILTERS[k]}"] = mag[:, k].astype(np.float32)
        c[f"err_{FILTERS[k]}"] = err[:, k].astype(np.float32)

    units = {"ra": "deg", "dec": "deg", "plx": "mas", "e_plx": "mas", "dist_pc": "pc"}
    for col in c.colnames:
        if col in units:
            c[col].unit = units[col]
        elif col.startswith(("mag_", "err_")) or col.startswith("gaia_g"):
            c[col].unit = "mag"
    c.meta["COMMENT"] = [
        "J-PLUS DR4 x Gaia DR3, d < 500 pc (plx > 2 mas), e_plx/plx < 0.1.",
        "J-PLUS mags: MAG_APER_COR_3_0 (AB), filters in wavelength order.",
        "Per filter, NaN = fails FLAGS<4 / MASK_FLAGS=0 / NORM_WMAP_VAL>0.8.",
        "Rows: all four of g, r, i, J0660 pass those cuts; one row per Gaia source.",
        "dist_pc = 1000/plx. No extinction correction applied.",
    ]
    c.write(out, overwrite=True)
    print(f"[save] {len(c)} stars, {len(c.colnames)} columns -> {out}")


def colour(t, a, b):
    return t["mag"][:, FILTERS.index(a)] - t["mag"][:, FILTERS.index(b)]


def plot(t, out=OUT_PNG):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ink, muted, grid = "#1f2328", "#59636e", "#d8dee4"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    n = len(DIAGRAMS) + 1
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.8), constrained_layout=True)

    def panel(ax, x, y, xl, yl, xlim, ylim):
        good = np.isfinite(x) & np.isfinite(y)
        x, y = np.asarray(x)[good], np.asarray(y)[good]
        # every star as a faint dot, density shading where the locus is crowded
        ax.scatter(x, y, s=1.5, c=ink, alpha=0.35, lw=0, rasterized=True)
        hb = ax.hexbin(x, y, gridsize=110, extent=(*xlim, *sorted(ylim)), mincnt=8,
                       cmap="Blues", norm=LogNorm(), linewidths=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, color=grid, lw=0.6)
        ax.set_axisbelow(True)
        return hb

    lims = [((-0.7, 2.9), (-0.7, 2.1)), ((-0.7, 2.9), (-1.15, 0.45))]
    for ax, ((xa, xb), (ya, yb)), (xl, yl) in zip(axes, DIAGRAMS, lims):
        panel(ax, colour(t, xa, xb), colour(t, ya, yb),
              f"{xa} − {xb}", f"{ya} − {yb}", xl, yl)

    gi = colour(t, "gSDSS", "iSDSS")
    Mr = t["mag"][:, FILTERS.index("rSDSS")] + 5 * np.log10(np.asarray(t["plx"], float) / 100.0)
    hb = panel(axes[-1], gi, Mr, "gSDSS − iSDSS", r"$M_r$  (from Gaia parallax)",
               (-1.2, 4.5), (17.5, 1.5))
    cb = fig.colorbar(hb, ax=axes[-1], shrink=0.8, pad=0.02)
    cb.set_label("stars per hex cell", color=muted)

    fig.suptitle(f"J-PLUS DR4 × Gaia DR3 — {len(t):,} stars, d < {1000/PLX_MIN_MAS:.0f} pc, "
                 f"σϖ/ϖ < {PLX_REL_ERR_MAX:.0%};  MAG_APER_COR_3_0, "
                 f"FLAGS<{FLAGS_MAX}, MASK_FLAGS=0, NORM_WMAP_VAL>{WMAP_MIN}",
                 color=ink, fontsize=11)
    fig.savefig(out, dpi=200)
    print(f"[plot] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if args.replot:
        from astropy.table import Table
        t = Table.read(OUT_FITS)
    else:
        print(QUERY)
        t = fetch()
    t = clean(t)
    save_clean(t)
    plot(t)


if __name__ == "__main__":
    main()
