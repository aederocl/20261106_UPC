"""Step 5: how much redshift information is in the XP coefficients, beyond broadband colours?

Feature sets (all include G magnitude, which carries some z information via the luminosity function):
  A_colours   : BP-RP, BP-G, G-RP, G
  B_xp        : 110 flux-normalized XP coefficients, G
  B_xp_low    : only the first 20 BP + 20 RP coefficients (robustness to noisy high orders), G
Models: kNN, LightGBM, MLP.  5-fold CV, fixed seed, target log(1+z).
Benchmark C: Gaia DR3 QSOC redshift (evaluated on the subset where it exists).
Control: LightGBM on B_xp with shuffled z (must have no skill).
"""
import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from common import DATA, FEATURES, FIG, N_COEF, PRED, SEED, confusion_ratios, metrics

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def feature_sets(df):
    xp = [f"c_{b}_{k:02d}" for b in ("bp", "rp") for k in range(N_COEF)]
    xp_low = [f"c_{b}_{k:02d}" for b in ("bp", "rp") for k in range(20)]
    return {
        "A_colours": ["bp_rp", "bp_g", "g_rp", "phot_g_mean_mag"],
        "B_xp": xp + ["phot_g_mean_mag"],
        "B_xp_low": xp_low + ["phot_g_mean_mag"],
    }


def models(n_feat):
    return {
        "kNN": lambda: make_pipeline(StandardScaler(), *( [PCA(n_components=30, whiten=True, random_state=SEED)]
                                                         if n_feat > 30 else []),
                                     KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63,
                                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.7, random_state=SEED, verbose=-1),
        # quantile -> Gaussian scaling: flux-normalised coefficients are heavy-tailed, which broke
        # StandardScaler + MLP on the 3k subset (sigma_NMAD 0.21, no better than colours)
        "MLP": lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                     MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=1e-3,
                                                  early_stopping=True, max_iter=600, random_state=SEED)),
    }


def cv_predict(make, X, y, folds):
    pred = np.empty_like(y)
    for tr, te in folds:
        mdl = make()
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    return pred


def main():
    df = pd.read_parquet(FEATURES)
    z = df.z.to_numpy()
    y = np.log1p(z)
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(df))
    rows, preds = [], {"source_id": df.source_id, "z": z}

    for fs_name, cols in feature_sets(df).items():
        X = df[cols].to_numpy(float)
        for m_name, make in models(len(cols)).items():
            zp = np.expm1(cv_predict(make, X, y, folds))
            key = f"{fs_name}|{m_name}"
            preds[key] = zp
            rows.append({"features": fs_name, "model": m_name, **metrics(z, zp)})
            print(rows[-1], flush=True)

    # shuffled-z control
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y)
    X = df[feature_sets(df)["B_xp"]].to_numpy(float)
    zp = np.expm1(cv_predict(models(X.shape[1])["LightGBM"], X, y_shuf, folds))
    rows.append({"features": "B_xp (shuffled z control)", "model": "LightGBM", **metrics(np.expm1(y_shuf), zp)})
    print(rows[-1])

    # Gaia QSOC benchmark on its own subset, with our models on the same subset for a fair comparison
    has = df.redshift_qsoc.notna().to_numpy()
    if has.sum() > 50:
        rows.append({"features": "C_gaia_QSOC", "model": "Gaia DR3 QSOC", **metrics(z[has], df.redshift_qsoc[has])})
        for key in ("A_colours|LightGBM", "B_xp|LightGBM"):
            rows.append({"features": key.split("|")[0] + " (QSOC subset)", "model": "LightGBM",
                         **metrics(z[has], preds[key][has])})
        preds["C_gaia_QSOC"] = df.redshift_qsoc.to_numpy()

    res = pd.DataFrame(rows)
    res.to_csv(DATA / "metrics.csv", index=False)
    pd.DataFrame(preds).to_parquet(PRED)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n", res.round(4).to_string(index=False))

    plot_scatter(df, preds, res)
    plot_binned(df, preds)


def plot_scatter(df, preds, res):
    z = df.z.to_numpy()
    best_b = res[res.features == "B_xp"].sort_values("sigma_NMAD").iloc[0].model
    best_a = res[res.features == "A_colours"].sort_values("sigma_NMAD").iloc[0].model
    panels = [(f"A_colours|{best_a}", f"colours only ({best_a})"), (f"B_xp|{best_b}", f"XP coefficients ({best_b})")]
    if "C_gaia_QSOC" in preds:
        panels.append(("C_gaia_QSOC", "Gaia DR3 QSOC"))
    zmax = np.percentile(z, 99.8)
    fig, ax = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.2), squeeze=False)
    zz = np.linspace(0, zmax, 100)
    for a, (key, title) in zip(ax[0], panels):
        zp = np.asarray(preds[key], float)
        ok = np.isfinite(zp)
        a.hexbin(z[ok], zp[ok], gridsize=90, extent=(0, zmax, 0, zmax), bins="log", cmap="Greys", mincnt=1)
        for name, r in confusion_ratios():
            line = (1 + zz) * r - 1
            if np.any((line > 0) & (line < zmax)) and 0.55 < r < 1.8:
                a.plot(zz, line, lw=0.5, alpha=0.6, c="#c53030")
        a.plot(zz, zz, c="#2b6cb0", lw=0.8)
        m = res[(res.features + "|" + res.model) == key]
        m = m.iloc[0] if len(m) else pd.Series(metrics(z[ok], zp[ok]))
        a.set(xlim=(0, zmax), ylim=(0, zmax), xlabel="SDSS z", ylabel="predicted z",
              title=f"{title}\nσ_NMAD={m.sigma_NMAD:.3f}  outl(0.2)={m['outlier_0.2']:.1%}")
    fig.text(0.01, 0.01, "red: loci of line mis-identification (1+z_pred)/(1+z_true)=λ_a/λ_b", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "z_pred_vs_true.png", dpi=160)


def plot_binned(df, preds):
    z = df.z.to_numpy()
    keys = [k for k in preds if "|" in k and (k.endswith("LightGBM"))] + (["C_gaia_QSOC"] if "C_gaia_QSOC" in preds else [])
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    for col, (var, label) in enumerate([(z, "SDSS z"), (df.phot_g_mean_mag.to_numpy(), "Gaia G")]):
        edges = np.quantile(var, np.linspace(0, 1, 13))
        mid = 0.5 * (edges[1:] + edges[:-1])
        for k in keys:
            zp = np.asarray(preds[k], float)
            s, o = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (var >= lo) & (var <= hi) & np.isfinite(zp)
                mm = metrics(z[sel], zp[sel]) if sel.sum() > 10 else {"sigma_NMAD": np.nan, "outlier_0.2": np.nan}
                s.append(mm["sigma_NMAD"]); o.append(mm["outlier_0.2"])
            ax[0, col].plot(mid, s, "o-", ms=3, label=k)
            ax[1, col].plot(mid, o, "o-", ms=3, label=k)
        ax[0, col].set(ylabel="σ_NMAD", yscale="log")
        ax[1, col].set(ylabel="outlier fraction (|Δz|/(1+z)>0.2)", xlabel=label)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "metrics_binned.png")


if __name__ == "__main__":
    main()
