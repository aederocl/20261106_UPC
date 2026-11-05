"""Step 4: look at the signal before fitting anything.

  figures/redshift_ladder.png   calibrated spectra stacked in z bins, relative to the mean shape,
                                with emission-line tracks lambda_rest*(1+z) overlaid
  figures/coef_vs_z.png         |Spearman rho| and mutual information of each coefficient with z
  figures/pca_by_z.png          first PCs of the normalized coefficients, coloured by z
  figures/sample.png            z and G distributions of the matched XP sample
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from common import DATA, FEATURES, FIG, LINES, N_COEF, SEED, SPECTRA

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def sample_plot(df):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    ax[0].hist(df.z, bins=80, color="0.3")
    ax[0].set(xlabel="SDSS z", ylabel="N", title=f"N = {len(df):,}")
    ax[1].hist(df.phot_g_mean_mag, bins=60, color="0.3")
    ax[1].set(xlabel="Gaia G")
    ax[2].scatter(df.z, df.phot_g_mean_mag, s=1, alpha=0.3, c="0.2", rasterized=True)
    ax[2].set(xlabel="z", ylabel="G")
    ax[2].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "sample.png")


def ladder(spec):
    wl, flux, z = spec["wl"], spec["flux"], spec["z"]
    ok = np.all(np.isfinite(flux), axis=1)
    wl_ok = (wl > 345) & (wl < 1000)   # edges are poorly constrained
    flux, z = flux[ok][:, wl_ok], z[ok]
    wl = wl[wl_ok]
    norm = flux / np.median(flux, axis=1, keepdims=True)

    # equal-count bins in z, so every row of the image has the same S/N
    nb = int(np.clip(len(z) // 40, 30, 150))
    edges = np.quantile(z, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, nb - 1)
    stack = np.vstack([np.median(norm[idx == i], axis=0) for i in range(nb)])
    zc = np.array([np.median(z[idx == i]) for i in range(nb)])
    rel = stack / np.median(stack, axis=0, keepdims=True)   # divide out the mean observed shape

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 2]})
    lim = np.nanpercentile(np.abs(np.log(np.clip(rel, 1e-3, None))), 98)
    dw = wl[1] - wl[0]
    ax[0].imshow(np.log(np.clip(rel, 1e-3, None)), cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", aspect="auto",
                 extent=[wl[0] - dw / 2, wl[-1] + dw / 2, 0, nb], interpolation="nearest")
    for name, lam in LINES.items():
        track = lam * (1 + zc)
        inside = (track > wl[0]) & (track < wl[-1])
        if inside.sum() > 1:
            ax[0].plot(track[inside], np.arange(nb)[inside] + 0.5, "k--", lw=0.7)
            j = np.flatnonzero(inside)[len(np.flatnonzero(inside)) // 2]
            ax[0].text(track[j] + 4, j + 0.5, name, fontsize=8)
    ticks = np.linspace(0, nb - 1, 8).astype(int)
    ax[0].set_yticks(ticks + 0.5, [f"{zc[t]:.2f}" for t in ticks])
    ax[0].set(xlabel="observed wavelength [nm]", ylabel="SDSS z (equal-count bins)",
              title="Gaia XP quasar spectra, stacked by z, relative to the mean shape (log ratio)")

    for k in np.linspace(0, nb - 1, 6).astype(int):
        ax[1].plot(wl, stack[k] / np.median(stack[k]), label=f"z≈{zc[k]:.2f}")
    ax[1].set(xlabel="observed wavelength [nm]", ylabel="normalized flux", title="example stacks")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "redshift_ladder.png", dpi=160)


def coef_relevance(df):
    cols = [f"c_{b}_{k:02d}" for b in ("bp", "rp") for k in range(N_COEF)]
    X = df[cols].to_numpy()
    rho = np.array([spearmanr(X[:, j], df.z).statistic for j in range(X.shape[1])])
    sub = df.sample(min(len(df), 20000), random_state=SEED)
    mi = mutual_info_regression(sub[cols].to_numpy(), sub.z, random_state=SEED)
    # same for the broadband colours, as the reference level
    mi_col = mutual_info_regression(sub[["bp_rp", "bp_g", "g_rp"]].to_numpy(), sub.z, random_state=SEED)
    pd.DataFrame({"coef": cols, "spearman": rho, "mutual_info": mi}).to_csv(DATA / "coef_relevance.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    x = np.arange(len(cols))
    for a, vals, lab in ((ax[0], np.abs(rho), "|Spearman ρ| with z"), (ax[1], mi, "mutual information [nats]")):
        a.bar(x[:N_COEF], vals[:N_COEF], color="#2b6cb0", label="BP")
        a.bar(x[N_COEF:], vals[N_COEF:], color="#c53030", label="RP")
        a.set_ylabel(lab)
    for v, name in zip(mi_col, ["BP−RP", "BP−G", "G−RP"]):
        ax[1].axhline(v, ls=":", lw=0.8, c="k")
        ax[1].text(len(cols) - 1, v, name, fontsize=7, ha="right", va="bottom")
    ax[0].legend()
    ax[1].set_xticks(np.r_[0:N_COEF:5, N_COEF:2 * N_COEF:5], [f"{k}" for k in np.r_[0:N_COEF:5, 0:N_COEF:5]])
    ax[1].set_xlabel("Hermite coefficient index (BP | RP)")
    fig.tight_layout()
    fig.savefig(FIG / "coef_vs_z.png")
    order = np.argsort(mi)[::-1][:10]
    print("Top-10 coefficients by MI:", [(cols[j], round(mi[j], 3)) for j in order])
    print("MI of colours BP-RP, BP-G, G-RP:", mi_col.round(3))


def pca_plot(df):
    cols = [c for c in df if c.startswith("c_")]
    Z = PCA(n_components=6, random_state=SEED).fit_transform(StandardScaler().fit_transform(df[cols]))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    order = np.argsort(df.z.to_numpy())
    for a, (i, j) in zip(ax, [(0, 1), (1, 2), (2, 3)]):
        sc = a.scatter(Z[order, i], Z[order, j], c=df.z.to_numpy()[order], s=1.5, cmap="turbo", rasterized=True)
        a.set(xlabel=f"PC{i + 1}", ylabel=f"PC{j + 1}")
        lo, hi = np.percentile(Z[:, [i, j]], [0.5, 99.5], axis=0)
        a.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    fig.colorbar(sc, ax=ax, label="SDSS z")
    fig.savefig(FIG / "pca_by_z.png")


def main():
    df = pd.read_parquet(FEATURES)
    spec = np.load(SPECTRA)
    sample_plot(df)
    ladder(spec)
    coef_relevance(df)
    pca_plot(df)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
