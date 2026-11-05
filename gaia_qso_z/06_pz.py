"""Step 6: probabilistic redshifts p(z) from the XP coefficients, to expose line-confusion degeneracies.

MLP classifier over bins in log(1+z), 5-fold CV -> p(z) for every quasar.
  figures/pz_examples.png     p(z) for well-measured, bimodal and catastrophic cases
  figures/pz_calibration.png  PIT histogram + coverage, peak multiplicity
  data/pz_summary.parquet     mode, median, 68% interval, n_peaks, PIT per quasar
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from common import DATA, FEATURES, FIG, N_COEF, SEED, metrics

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
NBINS = 60


def main():
    df = pd.read_parquet(FEATURES)
    cols = [f"c_{b}_{k:02d}" for b in ("bp", "rp") for k in range(N_COEF)] + ["phot_g_mean_mag"]
    X = df[cols].to_numpy(float)
    z = df.z.to_numpy()
    u = np.log1p(z)
    edges = np.linspace(u.min() - 1e-6, u.max() + 1e-6, NBINS + 1)
    lab = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, NBINS - 1)
    # merge empty classes away: LightGBM needs every label to exist in training data
    present = np.unique(lab)
    remap = {c: i for i, c in enumerate(present)}
    y = np.array([remap[c] for c in lab])
    centers = np.expm1(0.5 * (edges[present] + edges[present + 1]))
    widths = np.expm1(edges[present + 1]) - np.expm1(edges[present])
    upper = np.expm1(edges[present + 1])   # the CDF evaluated at bin upper edges

    P = np.zeros((len(df), len(present)))
    strat = np.minimum(y, np.percentile(y, 98).astype(int))   # avoid tiny strata
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat)):
        # LightGBM multiclass diverged here (one-hot probabilities, 33% outliers); an MLP classifier
        # on quantile-scaled features (same scaling as the best regressor) trains stably with log-loss
        mdl = make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                            MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True,
                                          max_iter=400, random_state=SEED))
        mdl.fit(X[tr], y[tr])
        P[np.ix_(te, mdl[-1].classes_)] = mdl.predict_proba(X[te])
        print(f"fold {k + 1}/5 done", flush=True)

    pdf = P / widths            # density per unit z
    cdf = np.cumsum(P, axis=1)
    z_mode = centers[np.argmax(P, axis=1)]
    z_med = np.array([np.interp(0.5, c, upper) for c in cdf])
    lo = np.array([np.interp(0.16, c, upper) for c in cdf])
    hi = np.array([np.interp(0.84, c, upper) for c in cdf])
    pit = np.array([np.interp(zt, upper, c) for zt, c in zip(z, cdf)])
    n_peaks = np.array([len(find_peaks(np.r_[0, p, 0], height=0.1 * p.max(), prominence=0.05 * p.max())[0])
                        for p in P])
    summ = pd.DataFrame({"source_id": df.source_id, "z": z, "z_mode": z_mode, "z_median": z_med,
                         "z_lo68": lo, "z_hi68": hi, "pit": pit, "n_peaks": n_peaks})
    summ.to_parquet(DATA / "pz_summary.parquet")
    np.savez_compressed(DATA / "pz_grid.npz", centers=centers, P=P.astype(np.float32))

    print("mode   :", metrics(z, z_mode))
    print("median :", metrics(z, z_med))
    print("fraction with >1 significant peak:", np.mean(n_peaks > 1).round(3))
    cov = np.mean((z >= lo) & (z <= hi))
    print(f"68% interval coverage: {cov:.3f}")
    outl = np.abs(z_mode - z) / (1 + z) > 0.2
    if outl.any():
        # for catastrophic outliers, how much probability did the model put near the truth?
        near = np.array([P[i, np.abs(centers - z[i]) / (1 + z[i]) < 0.05].sum() for i in np.flatnonzero(outl)])
        print(f"catastrophic outliers (mode): {outl.mean():.3f}; median p(|z-z_true|<0.05(1+z)) for them: "
              f"{np.median(near):.2f}; >0.2 for {np.mean(near > 0.2):.1%}")

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].hist(pit, bins=20, color="0.3")
    ax[0].axhline(len(pit) / 20, c="r", lw=0.8)
    ax[0].set(xlabel="PIT", title="PIT (flat = calibrated)")
    ax[1].hist(n_peaks, bins=np.arange(0.5, 6.5), color="0.3")
    ax[1].set(xlabel="number of significant p(z) peaks")
    ax[2].hexbin(z, z_mode, gridsize=80, bins="log", cmap="Greys", mincnt=1)
    ax[2].plot([0, z.max()], [0, z.max()], c="#2b6cb0", lw=0.8)
    ax[2].set(xlabel="SDSS z", ylabel="p(z) mode")
    fig.tight_layout()
    fig.savefig(FIG / "pz_calibration.png")

    rng = np.random.default_rng(SEED)
    groups = {
        "single peak, correct": np.flatnonzero((n_peaks == 1) & ~outl),
        "multi-peak": np.flatnonzero(n_peaks > 1),
        "catastrophic (mode)": np.flatnonzero(outl),
    }
    fig, ax = plt.subplots(3, 4, figsize=(12, 7), sharex=True)
    for r, (name, idx) in enumerate(groups.items()):
        pick = rng.choice(idx, size=min(4, len(idx)), replace=False) if len(idx) else []
        for c, i in enumerate(pick):
            a = ax[r, c]
            a.plot(centers, pdf[i], c="k", lw=1)
            a.axvline(z[i], c="#c53030", lw=1)
            a.set_title(f"{name}  G={df.phot_g_mean_mag.iloc[i]:.1f}", fontsize=8)
        ax[r, 0].set_ylabel("p(z)")
    for a in ax[-1]:
        a.set_xlabel("z  (red: SDSS)")
    fig.tight_layout()
    fig.savefig(FIG / "pz_examples.png")


if __name__ == "__main__":
    main()
