# src/validation/leakage_guards.py
from __future__ import annotations

from typing import Generator
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TARGET_COL = "internet_users_share_pct"

# Columns that represent exact duplicates or deterministic transforms of the target
LEAKAGE_TARGET = {
    "internet_users_count",
    "internet_users_share_pct_ict",
}

# Non-predictor columns to drop from feature matrix X
NON_PREDICTOR_COLS = {
    "entity",
    "code",
    "is_aggregate",
    "region",
    "i1_cluster",
    "i2_traj_cluster",
    "q10_forecast",
    "q50_forecast",
    "q90_forecast",
    "pred_ridge_i1",
    "pred_rf_i1",
    "pred_gbm_i1",
    "pred_gbm_i2",
    "residual_i1",
    "residual_i2",
}

# Duplicate or count transforms of predictors that cause collinearity or distortion
LEAKAGE_PREDICTORS = {
    "broadband_penetration_per100",  # Exact duplicate of broadband_subs_per100
    "broadband_subs_count",
    "broadband_10mbps_plus_count",
    "broadband_2to10mbps_count",
    "broadband_256kbto2mbps_count",
}

# Standard World Bank 7-Region Classification Mapping by ISO3 Code
ISO3_TO_REGION: dict[str, str] = {
    # East Asia and Pacific (EAS)
    "ASM": "East Asia & Pacific", "AUS": "East Asia & Pacific", "BRN": "East Asia & Pacific",
    "CHN": "East Asia & Pacific", "FJI": "East Asia & Pacific", "FSM": "East Asia & Pacific",
    "GUM": "East Asia & Pacific", "HKG": "East Asia & Pacific", "IDN": "East Asia & Pacific",
    "JPN": "East Asia & Pacific", "KHM": "East Asia & Pacific", "KIR": "East Asia & Pacific",
    "KOR": "East Asia & Pacific", "LAO": "East Asia & Pacific", "MAC": "East Asia & Pacific",
    "MHL": "East Asia & Pacific", "MMR": "East Asia & Pacific", "MNG": "East Asia & Pacific",
    "MNP": "East Asia & Pacific", "MYS": "East Asia & Pacific", "NCL": "East Asia & Pacific",
    "NRU": "East Asia & Pacific", "NZL": "East Asia & Pacific", "PHL": "East Asia & Pacific",
    "PLW": "East Asia & Pacific", "PNG": "East Asia & Pacific", "PRK": "East Asia & Pacific",
    "PYF": "East Asia & Pacific", "SGP": "East Asia & Pacific", "SLB": "East Asia & Pacific",
    "THA": "East Asia & Pacific", "TLS": "East Asia & Pacific", "TON": "East Asia & Pacific",
    "TUV": "East Asia & Pacific", "TWN": "East Asia & Pacific", "VNM": "East Asia & Pacific",
    "VUT": "East Asia & Pacific", "WSM": "East Asia & Pacific",

    # Europe and Central Asia (ECS)
    "ALB": "Europe & Central Asia", "AND": "Europe & Central Asia", "ARM": "Europe & Central Asia",
    "AUT": "Europe & Central Asia", "AZE": "Europe & Central Asia", "BEL": "Europe & Central Asia",
    "BGR": "Europe & Central Asia", "BIH": "Europe & Central Asia", "BLR": "Europe & Central Asia",
    "CHE": "Europe & Central Asia", "CHI": "Europe & Central Asia", "CYP": "Europe & Central Asia",
    "CZE": "Europe & Central Asia", "DEU": "Europe & Central Asia", "DNK": "Europe & Central Asia",
    "ESP": "Europe & Central Asia", "EST": "Europe & Central Asia", "FIN": "Europe & Central Asia",
    "FRA": "Europe & Central Asia", "FRO": "Europe & Central Asia", "GBR": "Europe & Central Asia",
    "GEO": "Europe & Central Asia", "GIB": "Europe & Central Asia", "GRC": "Europe & Central Asia",
    "GRL": "Europe & Central Asia", "HRV": "Europe & Central Asia", "HUN": "Europe & Central Asia",
    "IMN": "Europe & Central Asia", "IRL": "Europe & Central Asia", "ISL": "Europe & Central Asia",
    "ITA": "Europe & Central Asia", "KAZ": "Europe & Central Asia", "KGZ": "Europe & Central Asia",
    "LIE": "Europe & Central Asia", "LTU": "Europe & Central Asia", "LUX": "Europe & Central Asia",
    "LVA": "Europe & Central Asia", "MCO": "Europe & Central Asia", "MDA": "Europe & Central Asia",
    "MKD": "Europe & Central Asia", "MNE": "Europe & Central Asia", "NLD": "Europe & Central Asia",
    "NOR": "Europe & Central Asia", "POL": "Europe & Central Asia", "PRT": "Europe & Central Asia",
    "ROU": "Europe & Central Asia", "RUS": "Europe & Central Asia", "SMR": "Europe & Central Asia",
    "SRB": "Europe & Central Asia", "SVK": "Europe & Central Asia", "SVN": "Europe & Central Asia",
    "SWE": "Europe & Central Asia", "TJK": "Europe & Central Asia", "TKM": "Europe & Central Asia",
    "TUR": "Europe & Central Asia", "UKR": "Europe & Central Asia", "UZB": "Europe & Central Asia",
    "OWID_KOS": "Europe & Central Asia",

    # Latin America and Caribbean (LCN)
    "ABW": "Latin America & Caribbean", "ARG": "Latin America & Caribbean", "ATG": "Latin America & Caribbean",
    "BHS": "Latin America & Caribbean", "BLZ": "Latin America & Caribbean", "BOL": "Latin America & Caribbean",
    "BRA": "Latin America & Caribbean", "BRB": "Latin America & Caribbean", "CHL": "Latin America & Caribbean",
    "COL": "Latin America & Caribbean", "CRI": "Latin America & Caribbean", "CUB": "Latin America & Caribbean",
    "CUW": "Latin America & Caribbean", "CYM": "Latin America & Caribbean", "DMA": "Latin America & Caribbean",
    "DOM": "Latin America & Caribbean", "ECU": "Latin America & Caribbean", "GRD": "Latin America & Caribbean",
    "GTM": "Latin America & Caribbean", "GUY": "Latin America & Caribbean", "HND": "Latin America & Caribbean",
    "HTI": "Latin America & Caribbean", "JAM": "Latin America & Caribbean", "KNA": "Latin America & Caribbean",
    "LCA": "Latin America & Caribbean", "MAF": "Latin America & Caribbean", "MEX": "Latin America & Caribbean",
    "NIC": "Latin America & Caribbean", "PAN": "Latin America & Caribbean", "PER": "Latin America & Caribbean",
    "PRI": "Latin America & Caribbean", "PRY": "Latin America & Caribbean", "SLV": "Latin America & Caribbean",
    "SUR": "Latin America & Caribbean", "SXM": "Latin America & Caribbean", "TCA": "Latin America & Caribbean",
    "TTO": "Latin America & Caribbean", "URY": "Latin America & Caribbean", "VCT": "Latin America & Caribbean",
    "VEN": "Latin America & Caribbean", "VGB": "Latin America & Caribbean", "VIR": "Latin America & Caribbean",
    "AIA": "Latin America & Caribbean", "BMU": "Latin America & Caribbean", "MSR": "Latin America & Caribbean",

    # Middle East and North Africa (MEA)
    "ARE": "Middle East & North Africa", "BHR": "Middle East & North Africa", "DJI": "Middle East & North Africa",
    "DZA": "Middle East & North Africa", "EGY": "Middle East & North Africa", "IRN": "Middle East & North Africa",
    "IRQ": "Middle East & North Africa", "ISR": "Middle East & North Africa", "JOR": "Middle East & North Africa",
    "KWT": "Middle East & North Africa", "LBN": "Middle East & North Africa", "LBY": "Middle East & North Africa",
    "MAR": "Middle East & North Africa", "MLT": "Middle East & North Africa", "OMN": "Middle East & North Africa",
    "PSE": "Middle East & North Africa", "QAT": "Middle East & North Africa", "SAU": "Middle East & North Africa",
    "SYR": "Middle East & North Africa", "TUN": "Middle East & North Africa", "YEM": "Middle East & North Africa",

    # North America (NAC)
    "CAN": "North America", "USA": "North America", "SPM": "North America",

    # South Asia (SAS)
    "AFG": "South Asia", "BGD": "South Asia", "BTN": "South Asia", "IND": "South Asia",
    "LKA": "South Asia", "MDV": "South Asia", "NPL": "South Asia", "PAK": "South Asia",

    # Sub-Saharan Africa (SSF)
    "AGO": "Sub-Saharan Africa", "BDI": "Sub-Saharan Africa", "BEN": "Sub-Saharan Africa",
    "BFA": "Sub-Saharan Africa", "BWA": "Sub-Saharan Africa", "CAF": "Sub-Saharan Africa",
    "CIV": "Sub-Saharan Africa", "CMR": "Sub-Saharan Africa", "COD": "Sub-Saharan Africa",
    "COG": "Sub-Saharan Africa", "COM": "Sub-Saharan Africa", "CPV": "Sub-Saharan Africa",
    "ERI": "Sub-Saharan Africa", "ETH": "Sub-Saharan Africa", "GAB": "Sub-Saharan Africa",
    "GHA": "Sub-Saharan Africa", "GIN": "Sub-Saharan Africa", "GMB": "Sub-Saharan Africa",
    "GNB": "Sub-Saharan Africa", "GNQ": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa",
    "LBR": "Sub-Saharan Africa", "LSO": "Sub-Saharan Africa", "MDG": "Sub-Saharan Africa",
    "MLI": "Sub-Saharan Africa", "MOZ": "Sub-Saharan Africa", "MRT": "Sub-Saharan Africa",
    "MUS": "Sub-Saharan Africa", "MWI": "Sub-Saharan Africa", "NAM": "Sub-Saharan Africa",
    "NER": "Sub-Saharan Africa", "NGA": "Sub-Saharan Africa", "RWA": "Sub-Saharan Africa",
    "SDN": "Sub-Saharan Africa", "SEN": "Sub-Saharan Africa", "SLE": "Sub-Saharan Africa",
    "SOM": "Sub-Saharan Africa", "SSD": "Sub-Saharan Africa", "STP": "Sub-Saharan Africa",
    "SWZ": "Sub-Saharan Africa", "SYC": "Sub-Saharan Africa", "TCD": "Sub-Saharan Africa",
    "TGO": "Sub-Saharan Africa", "TZA": "Sub-Saharan Africa", "UGA": "Sub-Saharan Africa",
    "ZAF": "Sub-Saharan Africa", "ZMB": "Sub-Saharan Africa", "ZWE": "Sub-Saharan Africa",
}


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map ISO3 code to World Bank region with fallback 'Other'."""
    out = df.copy()
    if "code" in out.columns:
        out["region"] = out["code"].map(ISO3_TO_REGION).fillna("Other")
    else:
        out["region"] = "Other"
    return out


def filter_country_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to valid non-aggregate country observations:
    1. is_aggregate == False
    2. Valid 3-letter ISO3 code (or Kosovo 'OWID_KOS' if present)
    3. Excludes regional and income aggregates
    """
    out = df.copy()
    if "is_aggregate" in out.columns:
        out = out[out["is_aggregate"] == False]
    
    if "code" in out.columns:
        valid_iso = out["code"].notna() & (
            (out["code"].astype(str).str.len() == 3) | (out["code"] == "OWID_KOS")
        )
        out = out[valid_iso]

    out = add_region_column(out)
    return out.reset_index(drop=True)


def get_leaked_columns(df: pd.DataFrame) -> list[str]:
    """Find any leakage, target, or duplicate columns in a feature matrix."""
    leaked = set()
    cols = set(df.columns)
    
    # Check exact forbidden columns
    leaked.update(cols.intersection(LEAKAGE_TARGET))
    leaked.update(cols.intersection(LEAKAGE_PREDICTORS))
    if TARGET_COL in cols:
        leaked.add(TARGET_COL)
        
    # Check pattern matches (e.g. broadband_*_count) or target-lag patterns
    for col in df.columns:
        col_low = col.lower()
        if col_low.startswith("broadband_") and col_low.endswith("_count"):
            leaked.add(col)
        # Any feature containing 'internet' or 'target' is strictly forbidden in X
        if "internet" in col_low or "target" in col_low:
            leaked.add(col)
            
    return sorted(leaked)


def assert_no_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_years: pd.Series | None = None,
    test_years: pd.Series | None = None,
    train_groups: pd.Series | None = None,
    test_groups: pd.Series | None = None,
) -> None:
    """Strictly assert no feature leakage, no target leakage, and no split leakage."""
    # 1. Feature matrix leakage checks
    leaked_train = get_leaked_columns(X_train)
    leaked_test = get_leaked_columns(X_test)
    if leaked_train or leaked_test:
        raise ValueError(f"CRITICAL DATA LEAKAGE detected in features: {sorted(set(leaked_train + leaked_test))}")

    # 2. Target presence check
    if TARGET_COL in X_train.columns or TARGET_COL in X_test.columns:
        raise ValueError(f"CRITICAL TARGET LEAKAGE: '{TARGET_COL}' is present in feature matrix X.")

    # 3. Temporal holdout leakage check (train max < test min)
    if train_years is not None and test_years is not None and len(train_years) > 0 and len(test_years) > 0:
        if train_years.max() >= test_years.min():
            raise ValueError(
                f"TEMPORAL LEAKAGE: max train year ({train_years.max()}) >= min test year ({test_years.min()})."
            )

    # 4. Spatial / Group CV leakage check (disjoint entity sets)
    if train_groups is not None and test_groups is not None:
        overlap = set(train_groups.dropna()).intersection(set(test_groups.dropna()))
        if overlap:
            raise ValueError(
                f"SPATIAL / GROUP LEAKAGE: entity overlap across train and test folds: {sorted(overlap)[:5]}"
            )


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    allow_gdp: bool = False,
) -> pd.DataFrame:
    """Extract features while enforcing removal of leakage and non-predictor metadata."""
    cols = [c for c in feature_cols if c in df.columns]
    if not allow_gdp and "log_gdp_pc" in cols:
        cols.remove("log_gdp_pc")
        
    X = df[cols].copy()
    
    # Remove any forbidden columns
    forbidden = get_leaked_columns(X)
    if forbidden:
        X = X.drop(columns=forbidden)
        
    # Remove any non-predictor metadata columns if accidentally included
    meta_drop = [c for c in X.columns if c in NON_PREDICTOR_COLS]
    if meta_drop:
        X = X.drop(columns=meta_drop)
        
    return X


def get_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "entity",
    feature_cols: list[str] | None = None,
) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    """Generate 5-fold GroupKFold splits grouped by entity with leakage verification."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].astype(str)
    
    # Check that is_aggregate == True has been filtered out
    if "is_aggregate" in df.columns:
        assert (df["is_aggregate"] == True).sum() == 0, "Aggregate rows found in split dataset!"
        
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, df[TARGET_COL], groups=groups)):
        if feature_cols is not None:
            X_tr = build_feature_matrix(df.iloc[tr_idx], feature_cols)
            X_va = build_feature_matrix(df.iloc[va_idx], feature_cols)
            assert_no_leakage(
                X_tr,
                X_va,
                train_groups=groups.iloc[tr_idx],
                test_groups=groups.iloc[va_idx],
            )
        else:
            # Check groups overlap only
            overlap = set(groups.iloc[tr_idx].dropna()).intersection(set(groups.iloc[va_idx].dropna()))
            if overlap:
                raise ValueError(f"SPATIAL / GROUP LEAKAGE: entity overlap across train and test folds: {sorted(overlap)[:5]}")
        yield fold, tr_idx, va_idx
