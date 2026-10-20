"""
common.py - loaders, GDD and feature builders shared by all benchmark scripts.

Conventions
-----------
* DOY = calendar day of year (leap-year aware; matches the phenology sheets,
  e.g. 4 Sep 2012 = DOY 248).
* GDD = single-triangle growing degree days, base temperature T0 = 10 C.
* Windows [a, b] are inclusive at both ends.
* Precipitation: the meteo workbook only contains MONTHLY precipitation totals,
  so daily precipitation is approximated by spreading each monthly total
  evenly over the days of that month. Replace with NASA POWER daily data if
  you want the original thesis setup.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("OUT_DIR", "outputs"))

METEO_XLSX = "Date meteo_2010_2023.xlsx"
PHENO_XLSX = "Fenologie SCDVV Blaj 2010_2022.xlsx"
CHEM_XLSX = "Copy of date calitate struguri_SCDVV Blaj.xlsx"

T_BASE = 10.0
LAST_YEAR = 2022
PHASES = ["bb", "fl", "ve", "hv"]          # phase 1..4 of the paper
PHASE_LABELS = {"bb": "bud-break", "fl": "flowering", "ve": "veraison", "hv": "harvest onset"}
PROV_TO_VARIETY = {1: "Selena", 2: "Blasius", 3: "Radames",
                   4: "Sauvignon Blanc", 5: "Feteasca Alba"}
CHEM_KEYS = [("selena", "Selena"), ("blasius", "Blasius"), ("radames", "Radames"),
             ("sauvignon", "Sauvignon Blanc"), ("feteasca", "Feteasca Alba")]
GDD_MAX_DOY = 366


def out_path(*parts) -> Path:
    p = OUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _num(v) -> float:
    """Convert a cell to float; accepts comma decimals ('3,44') and strings."""
    if isinstance(v, str):
        v = v.strip().replace(",", ".")
        try:
            return float(v)
        except ValueError:
            return np.nan
    return float(v) if pd.notna(v) else np.nan


def gdd_single_triangle(tmax, tmin, tmed, t0: float = T_BASE) -> np.ndarray:
    """Single-triangle GDD (Sevacherian et al., 1977)."""
    tmax = np.asarray(tmax, float)
    tmin = np.asarray(tmin, float)
    tmed = np.asarray(tmed, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        triangle = (tmax - t0) ** 2 / (2.0 * (tmax - tmin))
    return np.where(tmax <= t0, 0.0,
                    np.where(tmin >= t0, tmed - t0, triangle))


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_weather(path=None):
    """
    Parse the daily meteo workbook (one sheet per year, 2010-2023).

    Each sheet has: a row per day ('ziua N'), 12 monthly blocks of 3 columns
    (Tmed, Tmax, Tmin), and one row 'Suma Precipitatii lunare' with monthly
    precipitation totals. Row positions differ between sheets (e.g. 2011, 2012
    have one row less at the top), so rows are located by their labels.

    Returns (daily_df, monthly_df).
    """
    path = Path(path) if path else DATA_DIR / METEO_XLSX
    xl = pd.ExcelFile(path)
    daily_rows, monthly_rows = [], []

    for sheet in xl.sheet_names:
        name = str(sheet).strip()
        if not name.isdigit():
            continue
        year = int(name)
        raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        labels = [str(v).strip().lower() for v in raw.iloc[:, 0].tolist()]

        # first occurrence of each 'ziua N' row only (ignores duplicated blocks)
        day_rows = {}
        for i, lab in enumerate(labels):
            m = re.fullmatch(r"ziua\s+(\d+)", lab)
            if m:
                day_rows.setdefault(int(m.group(1)), i)
        suma = [i for i, lab in enumerate(labels) if "suma" in lab and "precip" in lab]

        for month in range(1, 13):
            cols = [3 * (month - 1) + 1, 3 * (month - 1) + 2, 3 * (month - 1) + 3]
            p_month = np.nan
            if suma:
                vals = [_num(raw.iat[suma[0], c]) for c in cols if c < raw.shape[1]]
                vals = [v for v in vals if not np.isnan(v)]
                p_month = vals[0] if vals else np.nan
            ndays = pd.Timestamp(year, month, 1).days_in_month
            monthly_rows.append({"year": year, "month": month, "P_month_mm": p_month})

            for d, r in day_rows.items():
                try:
                    date = pd.Timestamp(year, month, d)
                except ValueError:
                    continue
                daily_rows.append({
                    "date": date, "year": year,
                    "Tmed": _num(raw.iat[r, cols[0]]),
                    "Tmax": _num(raw.iat[r, cols[1]]),
                    "Tmin": _num(raw.iat[r, cols[2]]),
                    "P_month_mm": p_month, "ndays": ndays,
                })

    daily = pd.DataFrame(daily_rows)
    daily = daily.dropna(subset=["Tmed", "Tmax", "Tmin"], how="all").copy()
    monthly = pd.DataFrame(monthly_rows)
    # missing monthly totals (e.g. May 2011) -> filled with the same-month mean, flagged
    monthly["P_imputed"] = monthly["P_month_mm"].isna()
    monthly["P_month_mm"] = monthly.groupby("month")["P_month_mm"].transform(
        lambda s: s.fillna(s.mean()))
    daily["month"] = daily["date"].dt.month
    daily = daily.drop(columns=["P_month_mm"]).merge(
        monthly[["year", "month", "P_month_mm"]], on=["year", "month"], how="left")
    daily["P"] = daily["P_month_mm"] / daily["ndays"]
    daily["doy"] = daily["date"].dt.dayofyear
    daily["tmed_imputed"] = daily["Tmed"].isna() & daily["Tmax"].notna() & daily["Tmin"].notna()
    daily["Tmed"] = daily["Tmed"].fillna((daily["Tmax"] + daily["Tmin"]) / 2)
    daily["GDD"] = gdd_single_triangle(daily["Tmax"], daily["Tmin"], daily["Tmed"])
    daily = daily.drop(columns=["ndays"]).sort_values("date").reset_index(drop=True)
    return daily, monthly


def load_phenology(path=None) -> pd.DataFrame:
    """Phenology DOYs per variety and season (Site 1 = SCDVV Blaj)."""
    path = Path(path) if path else DATA_DIR / PHENO_XLSX
    df = pd.read_excel(path, sheet_name=0).iloc[:, :7]
    df.columns = ["site", "prov", "year", "bb", "fl", "ve", "hv"]
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["variety"] = df["prov"].astype(int).map(PROV_TO_VARIETY)
    if df["variety"].isna().any():
        raise ValueError("Unknown province code in phenology file")
    return df[["variety", "year", "bb", "fl", "ve", "hv"]].reset_index(drop=True)


def load_chemistry(path=None) -> pd.DataFrame:
    """Sugar (g/L) and titratable acidity (g/L tartaric) at the analysis date."""
    path = Path(path) if path else DATA_DIR / CHEM_XLSX
    c = pd.read_excel(path, sheet_name=0).iloc[:, :6]
    c.columns = ["variety_raw", "year", "sample_date_raw", "sample_doy", "sugar_gL", "acid_gL"]
    c["variety_raw"] = c["variety_raw"].ffill()
    c = c.dropna(subset=["year"]).copy()
    c["year"] = c["year"].astype(int)
    c["variety"] = c["variety_raw"].astype(str).str.lower().map(
        lambda s: next((n for k, n in CHEM_KEYS if k in s), None))
    for col in ["sample_doy", "sugar_gL", "acid_gL"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    return c[["variety", "year", "sample_date_raw", "sample_doy", "sugar_gL", "acid_gL"]]\
        .reset_index(drop=True)


def weather_by_year(daily: pd.DataFrame, max_year: int = LAST_YEAR) -> dict:
    """{year: DataFrame indexed by DOY with Tmed, Tmax, Tmin, P, GDD}."""
    d = daily[daily.year <= max_year]
    return {int(y): g.drop_duplicates("doy").set_index("doy").sort_index()
            for y, g in d.groupby("year")}


# --------------------------------------------------------------------------- #
# Window helpers and features
# --------------------------------------------------------------------------- #
def window(series: pd.Series, a, b, how: str = "sum", min_cov: float = 0.9) -> float:
    """Sum or mean of a DOY-indexed series over [a, b]. NaN if coverage < min_cov."""
    idx = np.arange(int(a), int(b) + 1)
    v = series.reindex(idx)
    if v.notna().mean() < min_cov:
        return np.nan
    return float(v.sum()) if how == "sum" else float(v.mean())


def cutoff_features(wx_y: pd.DataFrame, phase_doys: dict) -> dict:
    """
    Features known at an information cutoff = the last observed phase.
    phase_doys: ordered {phase_name: DOY} for the phases observed so far.
    Uses only weather up to the cutoff (no target leakage).
    """
    names = list(phase_doys)
    doys = [float(phase_doys[n]) for n in names]
    f = {f"doy_{n}": d for n, d in zip(names, doys)}
    spans = [(f"jan1_{names[-1]}", 1, doys[-1])]
    spans += [(f"{names[i]}_{names[i + 1]}", doys[i], doys[i + 1])
              for i in range(len(doys) - 1)]
    for label, a, b in spans:
        f[f"gdd_{label}"] = window(wx_y["GDD"], a, b, "sum")
        f[f"prec_{label}"] = window(wx_y["P"], a, b, "sum")
        f[f"tmed_{label}"] = window(wx_y["Tmed"], a, b, "mean")
    return f


def bb_features(wx_y: pd.DataFrame, cutoff_doy: int) -> dict:
    """Weather known on 31 March (before any bud-break in 2010-2022)."""
    return {
        "gdd_jan1_cut": window(wx_y["GDD"], 1, cutoff_doy, "sum"),
        "tmed_jan1_cut": window(wx_y["Tmed"], 1, cutoff_doy, "mean"),
        "prec_jan1_cut": window(wx_y["P"], 1, cutoff_doy, "sum"),
    }