"""
02_benchmark.py - leave-one-year-out (LOYO) benchmark for bud-break and harvest onset.

Protocol (per variety, one fold per season):
  * train on all other seasons of the same variety, predict the held-out season;
  * every choice that depends on the data (feature selection, GDD start date,
    GDD threshold, climatology, scaler, network weights) is made on training
    seasons only;
  * features for a harvest forecast at cutoff k only use phases <= k and weather
    up to the observed date of phase k.

Models
  naive_mean          mean harvest/bud-break DOY of training seasons (paper's reference)
  ridge_doy           ridge on the observed phase DOYs known at the cutoff
  ridge_corr_sel      ridge on features with |r| >= 0.3 with the target (selected in-fold)
  mlp_corr_sel_eN     the paper's 32-16 ANN on the same selected features, N epochs,
                      averaged over 5 seeds
  gdd_forecast        GDD threshold model driven by training-year climatology
                      (valid forecast: no weather after the cutoff is used)
  gdd_retro           GDD threshold model driven by the observed weather of the test
                      season (retrospective reference, NOT a forecast; 366 = threshold
                      never reached in that season, see n_capped_366)
  ridge_weather       bud-break only: ridge on weather accumulated to 31 March

Outputs: outputs/tables/loyo_predictions.csv and outputs/tables/benchmark_summary.csv
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import (GDD_MAX_DOY, PHASES, bb_features, cutoff_features, load_phenology,
                    load_weather, out_path, weather_by_year, window)

RNG = np.random.default_rng(42)
N_BOOT = 2000
CORR_THR = 0.3
MLP_EPOCHS = [100, 1000]        # paper used 100 epochs (Keras); 1000 as a sensitivity check
SEEDS = [0, 1, 2, 3, 4]
BB_CUTOFF = 90                  # 31 March: every bud-break in 2010-2022 is later
HV_CUTOFFS = [0, 1, 2]          # cutoff after bud-break, flowering, veraison
BB_START_LOG = []               # selected GDD start per fold (reported)


# --------------------------------------------------------------------------- #
# GDD threshold models
# --------------------------------------------------------------------------- #
def gdd_sum(series, a, b):
    return window(series, a, b, "sum")


def climatology(wx, years):
    """Mean daily GDD per DOY over the given years."""
    stack = pd.concat([wx[int(y)]["GDD"].rename(int(y)) for y in years], axis=1)
    return stack.mean(axis=1).reindex(range(1, GDD_MAX_DOY + 1))


def gdd_threshold_predict(train, test, wx, mode):
    """
    Threshold = mean observed GDD between `start` and `target` over training seasons.
    Prediction = first DOY on which the accumulated GDD from the test season's
    `start` reaches the threshold. mode='forecast' uses climatology (no current-season
    weather); mode='observed' uses the test season's own observed weather.
    """
    s = [gdd_sum(wx[int(y)]["GDD"], st, tg)
         for y, st, tg in zip(train["year"], train["start"], train["target"])]
    thr = np.nanmean(s)
    if np.isnan(thr):
        return np.nan
    if mode == "forecast":
        series = climatology(wx, train["year"].tolist())
    else:
        series = wx[int(test["year"])]["GDD"]
    start = int(test["start"])
    cum = series.reindex(range(start, GDD_MAX_DOY + 1)).fillna(0.0).cumsum()
    hit = cum.index[cum.values >= thr]
    return float(hit[0]) if len(hit) else float(GDD_MAX_DOY)


def select_bb_start(train, wx, candidates=range(1, BB_CUTOFF + 1)):
    """GDD start day that minimises the coefficient of variation of the
    accumulated GDD up to bud-break across training seasons (thesis criterion)."""
    best = None
    for s in candidates:
        acc = np.array([gdd_sum(wx[int(y)]["GDD"], s, int(b))
                        for y, b in zip(train["year"], train["target"])], float)
        acc = acc[~np.isnan(acc)]
        if len(acc) < 4 or acc.mean() <= 0:
            continue
        cv = acc.std(ddof=1) / acc.mean()
        if best is None or cv < best[0]:
            best = (cv, s)
    return best[1] if best else None


# --------------------------------------------------------------------------- #
# Feature-based models
# --------------------------------------------------------------------------- #
def feature_cols(df, prefixes=("doy_", "gdd_", "prec_", "tmed_")):
    return [c for c in df.columns if c.startswith(prefixes)]


def drop_nan_cols(train, cols):
    return [c for c in cols if train[c].notna().all()]


def corr_select(train, cols, thr=CORR_THR):
    """Keep features with |Pearson r| >= thr with the target, computed on training seasons only."""
    y = train["target"].values.astype(float)
    keep = []
    for c in cols:
        x = train[c].values.astype(float)
        if np.nanstd(x) == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if abs(r) >= thr:
            keep.append(c)
    return keep


def fit_ridge(train, test, cols, alpha=1.0):
    if not cols or test[cols].isna().any():
        return np.nan
    X = train[cols].values.astype(float)
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=alpha).fit(sc.transform(X), train["target"].values.astype(float))
    xt = sc.transform(test[cols].values.astype(float).reshape(1, -1))
    return float(m.predict(xt)[0])


def fit_mlp(train, test, cols, epochs):
    if not cols or test[cols].isna().any():
        return np.nan
    X = train[cols].values.astype(float)
    y = train["target"].values.astype(float)
    sc = StandardScaler().fit(X)
    mu, sd = y.mean(), (y.std() or 1.0)
    xt = sc.transform(test[cols].values.astype(float).reshape(1, -1))
    preds = []
    for seed in SEEDS:
        net = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                           learning_rate_init=1e-3, alpha=1e-3, max_iter=epochs,
                           batch_size=min(16, len(X)), random_state=seed)
        net.fit(sc.transform(X), (y - mu) / sd)
        preds.append(net.predict(xt)[0] * sd + mu)
    return float(np.mean(preds))


# --------------------------------------------------------------------------- #
# Tables of features per task
# --------------------------------------------------------------------------- #
def build_harvest_table(pheno, wx, k):
    known = PHASES[: k + 1]
    rows = []
    for _, r in pheno.iterrows():
        y = int(r["year"])
        if y not in wx or pd.isna(r["hv"]) or any(pd.isna(r[p]) for p in known):
            continue
        feats = cutoff_features(wx[y], {p: r[p] for p in known})
        rows.append({"variety": r["variety"], "year": y,
                     "start": float(r[known[-1]]), "target": float(r["hv"]), **feats})
    return pd.DataFrame(rows)


def build_bb_table(pheno, wx):
    rows = []
    for _, r in pheno.iterrows():
        y = int(r["year"])
        if y not in wx or pd.isna(r["bb"]):
            continue
        rows.append({"variety": r["variety"], "year": y, "start": np.nan,
                     "target": float(r["bb"]), **bb_features(wx[y], BB_CUTOFF)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Predictors (train, test) -> {model_name: prediction}
# --------------------------------------------------------------------------- #
def predict_harvest(train, test, wx):
    cols_all = drop_nan_cols(train, feature_cols(train))
    doy_cols = [c for c in cols_all if c.startswith("doy_")]
    sel = corr_select(train, cols_all) or doy_cols
    out = {"naive_mean": float(train["target"].mean())}
    out["ridge_doy"] = fit_ridge(train, test, doy_cols) if doy_cols else out["naive_mean"]
    out["ridge_corr_sel"] = fit_ridge(train, test, sel)
    for e in MLP_EPOCHS:
        out[f"mlp_corr_sel_e{e}"] = fit_mlp(train, test, sel, e)
    out["gdd_forecast"] = gdd_threshold_predict(train, test, wx, "forecast")
    out["gdd_retro"] = gdd_threshold_predict(train, test, wx, "observed")
    return out


def predict_budbreak(train, test, wx):
    s_star = select_bb_start(train, wx)
    BB_START_LOG.append((int(test["year"]), s_star))
    tr = train.copy()
    tr["start"] = s_star
    te = test.copy()
    te["start"] = s_star
    cols = drop_nan_cols(tr, feature_cols(tr))
    out = {"naive_mean": float(train["target"].mean())}
    out["gdd_forecast"] = gdd_threshold_predict(tr, te, wx, "forecast")
    out["gdd_retro"] = gdd_threshold_predict(tr, te, wx, "observed")
    out["ridge_weather"] = fit_ridge(tr, te, cols)
    return out


# --------------------------------------------------------------------------- #
# LOYO driver and summary statistics
# --------------------------------------------------------------------------- #
def loyo(table, predict_fn, target, cutoff, wx):
    recs = []
    if table.empty:
        return recs
    for var, tv in table.groupby("variety"):
        tv = tv.sort_values("year").reset_index(drop=True)
        for i in range(len(tv)):
            test = tv.iloc[i]
            train = tv.drop(index=i)
            if len(train) < 5:
                continue
            preds = predict_fn(train, test, wx)
            for model, pred in preds.items():
                recs.append({"target": target, "cutoff": cutoff, "variety": var,
                             "year": int(test["year"]), "model": model,
                             "pred": pred, "actual": float(test["target"]),
                             "start": float(test["start"]) if pd.notna(test["start"]) else np.nan})
    return recs


def boot_ci_mean(x, B=N_BOOT):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(preds):
    preds = preds.dropna(subset=["pred"]).copy()
    preds["vkey"] = preds["variety"]
    preds["abs_err"] = (preds["pred"] - preds["actual"]).abs()
    pooled = preds.assign(variety="ALL")
    both = pd.concat([preds, pooled], ignore_index=True)

    key = ["target", "cutoff", "vkey", "year"]
    naive = both[both["model"] == "naive_mean"][key + ["abs_err"]].rename(
        columns={"abs_err": "naive_abs"}).drop_duplicates(key)
    merged = both.merge(naive, on=key, how="left")
    merged["delta_vs_naive"] = merged["abs_err"] - merged["naive_abs"]

    rows = []
    for (t, c, m, v), g in merged.groupby(["target", "cutoff", "model", "variety"]):
        e = g["pred"] - g["actual"]
        lo, hi = boot_ci_mean(g["abs_err"])
        d = g["delta_vs_naive"].dropna()
        dlo, dhi = boot_ci_mean(d)
        q05, q95 = np.percentile(e, [5, 95])
        lead = float((g["actual"] - g["start"]).mean()) if t == "harvest" else np.nan
        rows.append({
            "target": t, "cutoff": c, "variety": v, "model": m, "n": len(g),
            "MAE": g["abs_err"].mean(), "MAE_CI_low": lo, "MAE_CI_high": hi,
            "RMSE": float(np.sqrt((e ** 2).mean())), "bias": e.mean(),
            "PI90_halfwidth": (q95 - q05) / 2,
            "dMAE_vs_naive": d.mean(), "dMAE_CI_low": dlo, "dMAE_CI_high": dhi,
            "lead_days": lead, "n_capped_366": int((g["pred"] >= GDD_MAX_DOY).sum()),
        })
    return pd.DataFrame(rows)


def main():
    daily, _ = load_weather()
    wx = weather_by_year(daily)
    pheno = load_phenology()

    recs = []
    bb_table = build_bb_table(pheno, wx)
    recs += loyo(bb_table, predict_budbreak, "bud-break", "31-Mar", wx)
    for k in HV_CUTOFFS:
        tab = build_harvest_table(pheno, wx, k)
        recs += loyo(tab, predict_harvest, "harvest", PHASES[k], wx)

    preds = pd.DataFrame(recs)
    preds.to_csv(out_path("tables", "loyo_predictions.csv"), index=False)
    summary = summarise(preds)
    summary.to_csv(out_path("tables", "benchmark_summary.csv"), index=False)

    print("GDD start (bud-break models) selected per held-out year:", BB_START_LOG)
    show = summary[summary["variety"] == "ALL"][["target", "cutoff", "model", "n", "MAE",
                                                 "MAE_CI_low", "MAE_CI_high", "RMSE", "bias",
                                                 "dMAE_vs_naive", "dMAE_CI_low", "dMAE_CI_high",
                                                 "lead_days", "n_capped_366"]]
    print("\nPOOLED RESULTS (days):")
    print(show.round(2).to_string(index=False))
    print(f"\nWritten to {out_path('tables')}")


if __name__ == "__main__":
    main()