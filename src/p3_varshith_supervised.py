"""
p3_varshith_supervised.py  -  Varshith M Gowda (supervised learning + forecasting)

Task: predict a country-year's internet-user share from infrastructure indicators only
(no target lags: "can infrastructure explain uptake?"), and forecast 2021-25 with intervals.

Iteration 1 (raw data, no cleaning, complete cases)        -> Ridge, Random Forest, HistGBM
Iteration 2 (feedback-transformed data: artefacts removed, 13 engineered features,
            MICE imputation fitted inside each training fold) -> HistGBM + ablations
Forecast    temporal holdout, quantile HistGBM (q10/q50/q90) + split-conformal (CQR)
            re-calibration, motivated by the V13 drift chart.

Evaluation design:
  * one fixed country -> fold map for every experiment; Iteration 1 vs 2 also scored on identical rows
  * trajectory clusters are fitted on the target's own curve, so they are NOT used as features
  * interval coverage is reported as measured, plus a conformalised (CQR) interval
  * regional RMSEs are computed on identical rows for both iterations (V10)
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ENGINEERED, OUT, PROC, SEED, TARGET, load_panel, transform_for_iteration2
from p3_leakage_guards import assert_no_leakage, get_leaked_columns

warnings.filterwarnings("ignore")

RAW_CORE = ["broadband_subs_per100", "mobile_subs_per100", "telephone_lines_per100", "year"]
MISSING_FLAGS = [f"{c}_missing" for c in ["broadband_subs_per100", "mobile_subs_per100",
                                          "telephone_lines_per100", "mobile_phone_ownership_share_pct"]]
ITER2_FEATURES = RAW_CORE + ["mobile_phone_ownership_share_pct"] + ENGINEERED + MISSING_FLAGS
N_FOLDS = 5


def hgb(**kw):
    p = dict(max_iter=300, learning_rate=0.06, max_depth=8, random_state=SEED)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def fold_map(codes: pd.Series) -> dict:
    """One fixed country -> fold assignment used by EVERY experiment (GroupKFold semantics)."""
    u = np.array(sorted(codes.unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(u)
    return {c: i % N_FOLDS for i, c in enumerate(u)}


def _metrics(y, p):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, p))), mae=float(mean_absolute_error(y, p)),
                r2=float(r2_score(y, p)))


def add_missing_flags(d: pd.DataFrame) -> pd.DataFrame:
    for c in MISSING_FLAGS:
        d[c] = d[c.replace("_missing", "")].isna().astype(float)
    return d


def cv_run(df, feats, make_model, folds, imputer=None, perm=False):
    """Grouped CV; imputer/scaler are fitted on the training fold only."""
    oof = np.full(len(df), np.nan)
    rows, perms = [], []
    fold = df["code"].map(folds).values
    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        Xtr, Xva = df.loc[tr, feats], df.loc[va, feats]
        assert_no_leakage(Xtr, Xva, train_groups=df.loc[tr, "code"], test_groups=df.loc[va, "code"])
        if imputer is not None:
            imp = imputer()
            Xtr = pd.DataFrame(imp.fit_transform(Xtr), columns=feats, index=Xtr.index)
            Xva = pd.DataFrame(imp.transform(Xva), columns=feats, index=Xva.index)
        m = make_model().fit(Xtr, df.loc[tr, TARGET])
        p = m.predict(Xva)
        oof[va] = p
        rows.append({"fold": k, **_metrics(df.loc[va, TARGET], p)})
        if perm:   # importance measured on the HELD-OUT fold, not in-sample
            pi = permutation_importance(m, Xva, df.loc[va, TARGET], n_repeats=8, random_state=SEED,
                                        scoring="r2", n_jobs=-1)
            perms.append(pd.Series(pi.importances_mean, index=feats))
    res = pd.DataFrame(rows)
    out = {"rmse_mean": res.rmse.mean(), "rmse_std": res.rmse.std(), "mae_mean": res.mae.mean(),
           "mae_std": res.mae.std(), "r2_mean": res.r2.mean(), "r2_std": res.r2.std(), "n_rows": int(len(df))}
    return oof, out, (pd.concat(perms, axis=1) if perms else None)


MICE = lambda: IterativeImputer(max_iter=10, random_state=SEED)   # noqa: E731
MEDIAN = lambda: SimpleImputer(strategy="median")                 # noqa: E731


def run_cv_experiments(raw: pd.DataFrame, clean: pd.DataFrame):
    folds = fold_map(raw.dropna(subset=[TARGET]).code)
    clean = add_missing_flags(clean.copy())
    assert not get_leaked_columns(clean[ITER2_FEATURES]), "leakage in ITER2_FEATURES"

    # ---------------- Iteration 1: raw complete cases --------------------------------
    cc = raw.dropna(subset=[TARGET] + RAW_CORE)
    # keep only rows whose target survives cleaning, so I1 and I2 can be scored on identical rows
    ok_keys = clean.loc[clean[TARGET].notna(), ["code", "year"]]
    cc = cc.merge(ok_keys, on=["code", "year"]).reset_index(drop=True)
    res, oof = {}, {}
    oof["ridge_i1"], res["I1 Ridge (raw, complete case)"], _ = cv_run(
        cc, RAW_CORE, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), folds)
    oof["rf_i1"], res["I1 Random Forest (raw, complete case)"], _ = cv_run(
        cc, RAW_CORE, lambda: RandomForestRegressor(300, max_depth=12, n_jobs=-1, random_state=SEED), folds)
    oof["gbm_i1"], res["I1 HistGBM (raw, complete case)"], perm_i1 = cv_run(
        cc, RAW_CORE, lambda: hgb(max_iter=150, max_depth=6, learning_rate=0.08), folds, perm=True)

    # ---------------- Iteration 2 on the SAME rows (apples to apples) -----------------
    key = ["code", "year"]
    cc2 = cc[key].merge(clean, on=key, how="left")
    oof["gbm_i2_cc"], res["I2 HistGBM (engineered + MICE), same rows as I1"], _ = cv_run(
        cc2, ITER2_FEATURES, hgb, folds, imputer=MICE)

    # ---------------- Iteration 2 on all rows with a target + ablations ----------------
    full = clean.dropna(subset=[TARGET]).reset_index(drop=True)
    _, res["Ablation A: raw features + MICE, all rows"], _ = cv_run(
        full, RAW_CORE, lambda: hgb(max_iter=150, max_depth=6, learning_rate=0.08), folds, imputer=MICE)
    _, res["Ablation B: engineered + median imputation, all rows"], _ = cv_run(
        full, ITER2_FEATURES, hgb, folds, imputer=MEDIAN)
    oof["gbm_i2_full"], res["I2 HistGBM (engineered + MICE), all rows"], _ = cv_run(
        full, ITER2_FEATURES, hgb, folds, imputer=MICE)

    cc = cc.assign(pred_ridge_i1=oof["ridge_i1"], pred_rf_i1=oof["rf_i1"], pred_gbm_i1=oof["gbm_i1"],
                   pred_gbm_i2_same_rows=oof["gbm_i2_cc"])
    cc["residual_i1"] = cc[TARGET] - cc.pred_gbm_i1
    cc["residual_i2_same_rows"] = cc[TARGET] - cc.pred_gbm_i2_same_rows
    full = full.assign(pred_gbm_i2=oof["gbm_i2_full"])
    full["residual_i2"] = full[TARGET] - full.pred_gbm_i2
    perm_summary = pd.DataFrame({"mean": perm_i1.mean(1), "std": perm_i1.std(1)}).sort_values("mean")
    return res, cc, full, perm_summary


# --------------------------------------------------------------------------------------
# Forecast: temporal holdout with quantile models + conformalised quantile regression
# --------------------------------------------------------------------------------------
def run_forecast(clean: pd.DataFrame, train_end=2020, cal_years=(2018, 2020), alpha=0.2):
    d = add_missing_flags(clean.copy()).dropna(subset=[TARGET])
    feats = ITER2_FEATURES
    tr = d[d.year <= train_end]
    te = d[(d.year > train_end) & (d.year <= 2025)].copy()
    assert_no_leakage(tr[feats], te[feats], train_years=tr.year, test_years=te.year)

    def fit_q(train):
        imp = MICE().fit(train[feats])
        X = pd.DataFrame(imp.transform(train[feats]), columns=feats)
        ms = {q: hgb(loss="quantile", quantile=q).fit(X, train[TARGET]) for q in (alpha / 2, 0.5, 1 - alpha / 2)}
        return imp, ms

    def pred(imp, ms, df):
        X = pd.DataFrame(imp.transform(df[feats]), columns=feats)
        return {q: m.predict(X) for q, m in ms.items()}

    # (1) plain quantile model trained on <= 2020
    imp, ms = fit_q(tr)
    p = pred(imp, ms, te)
    lo, md, hi = p[alpha / 2], p[0.5], p[1 - alpha / 2]
    te["q10"], te["q50"], te["q90"] = lo, md, hi
    cov_raw = float(np.mean((te[TARGET] >= lo) & (te[TARGET] <= hi)))

    # (2) CQR: fit on <= 2017, conformity scores on 2018-2020, margin added to (1)
    tr0 = d[d.year < cal_years[0]]
    cal = d[(d.year >= cal_years[0]) & (d.year <= cal_years[1])]
    imp0, ms0 = fit_q(tr0)
    pc = pred(imp0, ms0, cal)
    scores = np.maximum(pc[alpha / 2] - cal[TARGET].values, cal[TARGET].values - pc[1 - alpha / 2])
    n = len(scores)
    qhat = float(np.quantile(scores, min(1, np.ceil((n + 1) * (1 - alpha)) / n)))
    te["q10_cqr"], te["q90_cqr"] = np.clip(lo - qhat, 0, 100), np.clip(hi + qhat, 0, 100)
    cov_cqr = float(np.mean((te[TARGET] >= te.q10_cqr) & (te[TARGET] <= te.q90_cqr)))

    # coverage by year: does it decay with horizon (drift)?
    by_year = te.assign(in_raw=(te[TARGET] >= te.q10) & (te[TARGET] <= te.q90),
                        in_cqr=(te[TARGET] >= te.q10_cqr) & (te[TARGET] <= te.q90_cqr),
                        err=te[TARGET] - te.q50).groupby("year").agg(
        n=("in_raw", "size"), coverage_raw=("in_raw", "mean"), coverage_cqr=("in_cqr", "mean"),
        mean_signed_error=("err", "mean"))
    te["residual_forecast"] = te[TARGET] - te.q50
    m = {"train_rows": int(len(tr)), "test_rows": int(len(te)), **{f"holdout_{k}": v for k, v in _metrics(te[TARGET], md).items()},
         "coverage_80_quantile": cov_raw, "coverage_80_cqr": cov_cqr, "cqr_margin_pp": qhat,
         "mean_width_raw": float(np.mean(hi - lo)), "mean_width_cqr": float(np.mean(te.q90_cqr - te.q10_cqr)),
         "mean_signed_error": float(te.residual_forecast.mean()),
         "share_underpredicted": float((te.residual_forecast > 0).mean())}
    return te, m, by_year


def main():
    raw = load_panel()
    clean, arts = transform_for_iteration2(raw)
    print("[CV] running iteration 1, iteration 2 and ablations (same country folds) ...")
    res, cc, full, perm = run_cv_experiments(raw, clean)
    tab = pd.DataFrame(res).T
    print(tab.round(3).to_string())
    print("\n[forecast] temporal holdout <=2020 -> 2021-25 ...")
    te, fm, by_year = run_forecast(clean)
    print(json.dumps(fm, indent=1))
    print(by_year.round(3).to_string())

    # regional error on identical rows
    reg = cc.groupby("region").apply(lambda g: pd.Series({
        "n": len(g), "rmse_i1": np.sqrt(np.mean(g.residual_i1 ** 2)),
        "rmse_i2": np.sqrt(np.mean(g.residual_i2_same_rows ** 2)),
        "median_resid_i1": g.residual_i1.median(), "median_resid_i2": g.residual_i2_same_rows.median()}))
    print(reg.round(2).to_string())

    tab.to_csv(OUT / "p3_cv_results.csv")
    reg.to_csv(OUT / "p3_regional_rmse_same_rows.csv")
    perm.to_csv(OUT / "p3_iter1_permutation_importance_heldout.csv")
    by_year.to_csv(OUT / "p3_forecast_coverage_by_year.csv")
    cc.to_parquet(PROC / "p3_iter1_oof.parquet", index=False)
    full.to_parquet(PROC / "p3_iter2_oof.parquet", index=False)
    te.to_parquet(PROC / "p3_forecast_holdout.parquet", index=False)
    clean.to_parquet(PROC / "panel_iteration2.parquet", index=False)
    with open(OUT / "p3_model_metrics.json", "w") as f:
        json.dump({"cv": res, "forecast": fm, "regional_same_rows": reg.round(3).to_dict(orient="index")},
                  f, indent=2)
    return res, cc, full, perm, te, fm, by_year, reg


if __name__ == "__main__":
    main()
