"""
01_audit.py - load, clean and audit the three raw workbooks.

Writes:
  outputs/clean/*.csv            cleaned tables used by 02_benchmark.py
  outputs/audit/audit.txt        data facts to put in Chapters 3-4 of the paper

Checks the paper's numbers (Table I, annual means of Table 3.1) against the raw files.
"""
import numpy as np
import pandas as pd

from .common import (DATA_DIR, METEO_XLSX, PHASES, PHASE_LABELS, load_chemistry,
                     load_phenology, load_weather, out_path)

# Table I of the submitted paper: mean, sd (days), n seasons
PAPER_TABLE_I = {
    "Selena":          {"bb": (109, 6.5), "fl": (158, 5.1), "ve": (227, 6.4), "hv": (261, 7.5)},
    "Blasius":         {"bb": (110, 6.0), "fl": (158, 6.3), "ve": (228, 5.8), "hv": (262, 7.9)},
    "Radames":         {"bb": (110, 6.1), "fl": (161, 10.0), "ve": (230, 6.0), "hv": (265, 6.5)},
    "Sauvignon Blanc": {"bb": (111, 5.5), "fl": (160, 6.6), "ve": (230, 6.0), "hv": (265, 6.4)},
    "Feteasca Alba":   {"bb": (107, 6.7), "fl": (162, 13.4), "ve": (227, 6.2), "hv": (260, 8.0)},
}
PAPER_N = {"Selena": 11, "Blasius": 13, "Radames": 10, "Sauvignon Blanc": 13, "Feteasca Alba": 13}
PAPER_ANNUAL_MEAN_T = {2010: 10.0, 2011: 9.3, 2012: 10.6, 2013: 11.1, 2014: 11.8, 2015: 11.2,
                       2016: 10.7, 2017: 10.6, 2018: 11.8, 2019: 11.9, 2020: 10.8, 2021: 10.2,
                       2022: 10.9}
VARIETIES = list(PAPER_TABLE_I)

log_lines = []


def p(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    log_lines.append(s)


def check_extra_rows_after_precip(path):
    """Report rows after the precipitation row (e.g. comma-decimal duplicate blocks)."""
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        if not str(sheet).strip().isdigit():
            continue
        raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        labels = [str(v).strip().lower() for v in raw.iloc[:, 0]]
        suma = [i for i, l in enumerate(labels) if "suma" in l and "precip" in l]
        if not suma:
            p(f"  {sheet}: NO monthly precipitation row found")
            continue
        extra = raw.iloc[suma[0] + 1:].dropna(how="all")
        if len(extra) == 0:
            continue
        comma = sum(isinstance(v, str) and "," in v for v in extra.values.ravel())
        p(f"  {sheet}: {len(extra)} extra rows after precipitation row "
          f"({comma} comma-decimal strings) -> ignored")
        # is the extra block a copy of the daily block? compare first 31 rows
        day_vals = raw.iloc[suma[0] - 31:suma[0], 1:4].reset_index(drop=True)
        extra_vals = extra.iloc[:31, 1:4].reset_index(drop=True)
        if len(extra_vals) == len(day_vals) == 31:
            num_extra = extra_vals.map(lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan)
            num_day = day_vals.apply(pd.to_numeric, errors="coerce")
            same = np.allclose(num_extra.values.astype(float), num_day.values.astype(float),
                               equal_nan=True, atol=1e-6)
            p(f"      first 31 extra rows identical to daily block (cols Jan Tmed-Tmin)? {same}")


def main():
    # ----------------------------------------------------------------- weather
    daily, monthly = load_weather()
    pheno = load_phenology()
    chem = load_chemistry()

    daily.to_csv(out_path("clean", "common.csv"), index=False)
    monthly.to_csv(out_path("clean", "precip.csv"), index=False)
    pheno.to_csv(out_path("clean", "pheno.csv"), index=False)
    chem.to_csv(out_path("clean", "chemistry.csv"), index=False)

    p("=" * 78)
    p("1. WEATHER WORKBOOK", DATA_DIR / METEO_XLSX)
    p("=" * 78)
    p("Sheet-level layout check (extra rows / duplicated blocks):")
    check_extra_rows_after_precip(DATA_DIR / METEO_XLSX)

    yearly = daily.groupby("year").agg(
        days=("date", "size"),
        tmed_imputed=("tmed_imputed", "sum"),
        annual_mean_Tmed=("Tmed", "mean"),
    )
    viol = daily.assign(v=daily.Tmax < daily.Tmin).groupby("year")["v"].sum()
    yearly["tmax_lt_tmin"] = viol
    yearly["paper_annual_mean"] = pd.Series(PAPER_ANNUAL_MEAN_T)
    yearly["diff"] = yearly["annual_mean_Tmed"] - yearly["paper_annual_mean"]
    p("\nPer-year summary (annual mean of daily Tmed vs. paper Table 3.1):")
    p(yearly.round(2).to_string())
    p("\nMonths with monthly precipitation total present (of 12):")
    mp = monthly.groupby("year")["P_month_mm"].apply(lambda s: int(s.notna().sum()))
    p(mp.to_string())
    missing_precip = monthly[monthly.P_imputed]
    if len(missing_precip):
        p("Imputed monthly precipitation (same-month mean):", missing_precip[["year", "month"]].values.tolist())
    p("Note: daily precipitation is spread uniformly within each month (approximation).")
    p("Note: 2023 is incomplete (December missing) and is excluded from the benchmark.")

    # ----------------------------------------------------------- phenology
    p("\n" + "=" * 78)
    p("2. PHENOLOGY vs PAPER TABLE I (mean, sd with ddof=1, [min-max], n)")
    p("=" * 78)
    for v in VARIETIES:
        g = pheno[pheno.variety == v]
        for ph in PHASES:
            s = g[ph].dropna()
            pm, psd = PAPER_TABLE_I[v][ph]
            dm = s.mean() - pm
            ds = s.std(ddof=1) - psd
            flag = "" if (abs(dm) < 1.0 and abs(ds) < 0.5) else "   <-- differs from paper"
            p(f"{v:16s} {ph:3s} n={len(s):2d} mean={s.mean():6.1f} sd={s.std(ddof=1):5.1f} "
              f"[{s.min():.0f}-{s.max():.0f}] | paper {pm}±{psd}{flag}")
        p(f"{'':16s} n seasons: data={len(g)}  paper={PAPER_N[v]}"
          + ("   <-- paper n differs" if len(g) != PAPER_N[v] else ""))

    order_bad = pheno[~((pheno.bb < pheno.fl) & (pheno.fl < pheno.ve) & (pheno.ve < pheno.hv))]
    p(f"\nPhase order violations (bb<fl<ve<hv): {len(order_bad)}")
    p("Correlation of phases with harvest DOY (all varieties pooled):")
    p(pheno[["bb", "fl", "ve", "hv"]].corr()["hv"].round(2).to_string())

    # ----------------------------------------------------------- chemistry
    p("\n" + "=" * 78)
    p("3. CHEMISTRY (single analysis per variety and season)")
    p("=" * 78)
    p("Rows per variety (non-missing sugar):")
    p(chem.dropna(subset=["sugar_gL"]).groupby("variety").size().to_string())
    miss = chem[chem.sugar_gL.isna() | chem.acid_gL.isna()][["variety", "year"]]
    p("Missing sugar or acidity:", miss.values.tolist())
    p(f"Sugar range g/L: {chem.sugar_gL.min():.1f} - {chem.sugar_gL.max():.1f} "
      f"(values >220 g/L or <120 g/L should be checked against the lab sheets)")
    p("Suspicious sugar values:",
      chem[(chem.sugar_gL > 220) | (chem.sugar_gL < 120)][["variety", "year", "sugar_gL"]].values.tolist())
    p("Raw analysis-date formats present (mixed text such as '25.IX', '04.X', '14.x'):")
    p(chem.sample_date_raw.astype(str).str.replace(r"\d", "#", regex=True).value_counts().to_string())
    both = chem.merge(pheno, on=["variety", "year"], how="inner").dropna(subset=["sample_doy", "hv"])
    off = both.sample_doy - both.hv
    p(f"\nAnalysis DOY minus phenological harvest DOY: mean={off.mean():.1f} d, "
      f"sd={off.std():.1f} d, n={len(off)}  -> the analysis is NOT the same event as Date4; "
      f"document this in the paper.")
    if len(both) > 3:
        p(f"Pearson r(sugar, harvest DOY) pooled = {both.sugar_gL.corr(both.hv):.2f}; "
          f"r(acid, harvest DOY) pooled = {both.acid_gL.corr(both.hv):.2f}")

    p("\nPaper caveats to confirm before submission:")
    p("  - Table I n for Selena is 11 in the paper but 13 in the file.")
    p("  - Chapter 4 bud-break ANN uses cumulative heat UP TO the observed bud-break date "
      "as input (target leakage in a forecasting sense).")
    p("  - Harvest-model feature selection (|r|>=0.3) was run on the full series (leakage).")
    p("  - Yield data are not in the uploaded files; the benchmark covers phenology only.")

    out_path("audit").mkdir(parents=True, exist_ok=True)
    (out_path("audit") / "audit.txt").write_text("\n".join(log_lines), encoding="utf-8")
    p(f"\nWritten: {out_path('audit') / 'audit.txt'}")


if __name__ == "__main__":
    main()