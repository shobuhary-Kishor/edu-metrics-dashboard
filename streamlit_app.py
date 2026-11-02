"""
Zone Marks Analysis — Streamlit App
===================================
Streamlit version of the Zone Marks Analysis module (v1.4.3, MBM.Kishor).

Combines two analyses on one uploaded Excel file:
  * Structure 1 – student-level marks analysis (pass rates, remedial targets,
    descriptive statistics, normality tests, 25+ graphs)
  * Structure 2 – exam-department school / zone performance summaries

Run with:   streamlit run streamlit_app.py
"""

from scipy.stats import (
    anderson, kstest, norm, normaltest, pearsonr, probplot, shapiro, zscore,
)
from scipy.interpolate import make_interp_spline
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile

import matplotlib
matplotlib.use("Agg")

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10

COLUMN_MAP = {
    "No": "Index",
    "ZONE": "Zone",
    "Name of School": "Name_of_School",
    "Name of Students": "Name_of_Students",
    "Part-I": "Part_I",
    "Part - II": "Part_II",
    "Total": "Total",
}
REQUIRED_RAW = ["ZONE", "Name of School", "Name of Students",
                "Part-I", "Part - II", "Total"]


# =====================================================================
# Helpers
# =====================================================================
def fig_to_png(fig, dpi=130) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)


def missing_columns(raw: pd.DataFrame):
    return [c for c in REQUIRED_RAW if c not in raw.columns]


# =====================================================================
# STRUCTURE 1 – Marks analysis
# =====================================================================
def clean_and_analyze(raw: pd.DataFrame, target_percentage: float):
    df = raw.rename(columns=COLUMN_MAP).copy()

    df[["Part_I", "Part_II"]] = df[["Part_I", "Part_II"]].fillna(0)
    df["Check_Total"] = df["Part_I"] + df["Part_II"]
    false_records = df[df["Check_Total"] != df["Total"]].copy()

    df["Average"] = (df["Part_I"] + df["Part_II"]) / 2
    df["P1_above_35"] = np.where(df["Part_I"] >= 35, "QP1", "NQP1")
    df["P2_above_35"] = np.where(df["Part_II"] >= 35, "QP2", "NQP2")
    df["above_70"] = np.where((df["Part_I"] >= 35) &
                              (df["Part_II"] >= 35), "Q", "NQ")
    df["Diff_from_35"] = df["Average"] - 35

    mean_score = df["Average"].mean()
    std_score = df["Average"].std(ddof=0)
    if std_score == 0 or np.isnan(std_score):
        std_score = 1
    df["Z_Score"] = (df["Average"] - mean_score) / std_score
    df["T_Score"] = 50 + 10 * df["Z_Score"]

    target_students = df[df["Average"] < 35].copy()

    school_total = df.groupby("Name_of_School")["Average"].count()
    school_pass = (
        df.assign(_pass=(df["Average"] > 35).astype(int))
        .groupby("Name_of_School")["_pass"].sum()
    )
    school_summary = pd.DataFrame({
        "Total_Students": school_total,
        "Passed": school_pass,
        "Current_%": (school_pass / school_total) * 100,
    })

    total_students = len(df)
    passed_students = int((df["Average"] > 35).sum())
    target_passes = int((target_percentage / 100) * total_students)
    needed_passes = target_passes - passed_students

    school_summary["Target_Passes"] = (
        school_summary["Total_Students"] * target_percentage / 100
    ).round().astype(int)
    school_summary["Needed_Passes"] = (
        school_summary["Target_Passes"] - school_summary["Passed"]
    ).clip(lower=0)

    return df, false_records, needed_passes, school_summary, target_students


def advanced_statistics(df: pd.DataFrame, score_column="Average"):
    """Returns tables plus a context dict used by the plotting functions."""
    a = (df[["Name_of_Students", score_column]].dropna(subset=[score_column]).copy()
         .rename(columns={"Name_of_Students": "Student", score_column: "Score"}))
    raw_scores = a["Score"].tolist()
    n = len(raw_scores)
    if n == 0:
        return None

    a["Rank"] = a["Score"].rank(ascending=False, method="min").astype(int)
    a["Z_Score"] = zscore(a["Score"])
    a["T_Score"] = 50 + 10 * a["Z_Score"]
    a["Percentile"] = a["Score"].rank(pct=True) * 100
    a["Deviation"] = a["Score"] - a["Score"].mean()
    a["Squared_Deviation"] = a["Deviation"] ** 2

    # Frequency distribution (Sturges)
    k = int(np.ceil(1 + 3.322 * np.log10(n))) if n > 1 else 1
    data_min, data_max = min(raw_scores), max(raw_scores)
    R = data_max - data_min
    class_width = int(np.ceil(R / k)) if R > 0 else 1
    bins = np.arange(data_min, data_max + class_width + 1, class_width)
    if len(bins) < 2:
        bins = np.array([data_min, data_min + 1])
    freq_table = pd.cut(a["Score"], bins=bins,
                        right=False).value_counts().sort_index()

    freq_df = pd.DataFrame()
    if len(freq_table) > 0:
        freq_df = pd.DataFrame({
            "Class Interval": [f"{int(iv.left)} - {int(iv.right - 1)}" for iv in freq_table.index],
            "Frequency (f)": freq_table.values,
            "Midpoint (x)": [(iv.left + iv.right) / 2 for iv in freq_table.index],
            "Cumulative f": freq_table.values.cumsum(),
            "Relative f": freq_table.values / n,
            "Rel. Cum. f": (freq_table.values / n).cumsum(),
        })
        freq_df["f * x"] = freq_df["Frequency (f)"] * freq_df["Midpoint (x)"]

    mean_val = float(np.mean(raw_scores))
    median_val = float(np.median(raw_scores))
    mode_val = a["Score"].mode().tolist()
    std_sample = float(np.std(raw_scores, ddof=1)) if n > 1 else 0.0
    std_pop = float(np.std(raw_scores, ddof=0))
    variance = float(np.var(raw_scores, ddof=1)) if n > 1 else 0.0
    q1, q3 = np.percentile(raw_scores, [25, 75])
    iqr = q3 - q1
    skewness = float(a["Score"].skew())
    kurt = float(a["Score"].kurtosis())

    stats_summary = pd.DataFrame({
        "Statistic": ["N", "Mean", "Median", "Mode(s)", "Sample Std Dev",
                      "Population Std Dev", "Variance", "Minimum", "Maximum",
                      "Range", "Q1", "Q3", "IQR", "Skewness", "Kurtosis"],
        "Value": [n, round(mean_val, 3), median_val, str(mode_val),
                  round(std_sample, 3), round(std_pop, 3), round(variance, 3),
                  data_min, data_max, R, round(q1, 3), round(q3, 3),
                  round(iqr, 3), round(skewness, 3), round(kurt, 3)],
    })

    score_table = (a[["Student", "Score", "Z_Score", "T_Score", "Percentile"]]
                   .sort_values("Score", ascending=False).reset_index(drop=True))

    corr_matrix = a[["Score", "Z_Score",
                     "T_Score", "Percentile", "Rank"]].corr()
    r_score_tscore, _ = pearsonr(a["Score"], a["T_Score"])

    def _z(x):
        return (x - mean_val) / std_sample if std_sample > 0 else np.nan

    ok = std_sample > 0
    prob_df = pd.DataFrame({
        "Interval": ["Score < 50", "Score ≥ 60", "50 ≤ Score ≤ 70",
                     "Within 1 SD (±1σ)", "Within 2 SD (±2σ)"],
        "Z-Interval": [
            f"z < {_z(50):.3f}" if ok else "N/A",
            f"z ≥ {_z(60):.3f}" if ok else "N/A",
            f"{_z(50):.3f} ≤ z ≤ {_z(70):.3f}" if ok else "N/A",
            "-1 ≤ z ≤ 1", "-2 ≤ z ≤ 2"],
        "Theoretical P": [
            round(norm.cdf(50, mean_val, std_sample), 4) if ok else 0.0,
            round(1 - norm.cdf(60, mean_val, std_sample), 4) if ok else 0.0,
            round(norm.cdf(70, mean_val, std_sample) -
                  norm.cdf(50, mean_val, std_sample), 4) if ok else 0.0,
            round(norm.cdf(1) - norm.cdf(-1), 4),
            round(norm.cdf(2) - norm.cdf(-2), 4)],
        "Empirical Count": [
            int((a["Score"] < 50).sum()),
            int((a["Score"] >= 60).sum()),
            int(((a["Score"] >= 50) & (a["Score"] <= 70)).sum()),
            int(((a["Z_Score"] >= -1) & (a["Z_Score"] <= 1)).sum()),
            int(((a["Z_Score"] >= -2) & (a["Z_Score"] <= 2)).sum())],
    })
    prob_df["Empirical P"] = (prob_df["Empirical Count"] / n).round(4)

    # Normality tests
    if n >= 8:
        sh_stat, sh_p = shapiro(a["Score"])
        ks_stat, ks_p = kstest(a["Score"], norm.cdf,
                               args=(mean_val, std_sample))
        try:
            dag_stat, dag_p = normaltest(a["Score"])
        except ValueError:
            dag_stat, dag_p = np.nan, np.nan
        try:
            ad = anderson(a["Score"], dist="norm", method="interpolate")
            ad_p = float(getattr(ad, "pvalue", np.nan))
            ad_uses_p = not np.isnan(ad_p)
        except TypeError:
            ad = anderson(a["Score"], dist="norm")
            ad_p, ad_uses_p = None, False
        if ad_uses_p:
            ad_cell = round(ad_p, 4)
            ad_concl = "Normal" if ad_p > 0.05 else "Not Normal"
        else:
            ad_cell = f"{ad.critical_values[2]:.4f} (crit@5%)"
            ad_concl = "Normal" if ad.statistic < ad.critical_values[2] else "Not Normal"
        normality = pd.DataFrame({
            "Test": ["Shapiro-Wilk", "Kolmogorov-Smirnov", "D'Agostino-Pearson", "Anderson-Darling"],
            "Statistic": [round(sh_stat, 4), round(ks_stat, 4),
                          round(dag_stat, 4) if not np.isnan(
                              dag_stat) else "N/A",
                          round(ad.statistic, 4)],
            "p-value": [round(sh_p, 4), round(ks_p, 4),
                        round(dag_p, 4) if not np.isnan(dag_p) else "N/A", ad_cell],
            "Conclusion (α=0.05)": [
                "Normal" if sh_p > 0.05 else "Not Normal",
                "Normal" if ks_p > 0.05 else "Not Normal",
                "Normal" if (not np.isnan(dag_p) and dag_p >
                             0.05) else "Not Normal",
                ad_concl],
        })
    else:
        normality = pd.DataFrame({
            "Test": ["Insufficient Data (n<8)"], "Statistic": [np.nan],
            "p-value": [np.nan], "Conclusion (α=0.05)": ["N/A"]})

    ctx = dict(a=a, freq_df=freq_df, freq_table=freq_table, bins=bins,
               class_width=class_width, mean_val=mean_val, median_val=median_val,
               std_sample=std_sample, data_min=data_min, data_max=data_max, n=n,
               corr_matrix=corr_matrix, r=r_score_tscore)
    return dict(analysis_df=a, freq_df=freq_df, stats_summary=stats_summary,
                score_table=score_table, corr_matrix=corr_matrix, prob_df=prob_df,
                normality=normality, ctx=ctx)


# =====================================================================
# STRUCTURE 2 – Exam department
# =====================================================================
def _perf_table(grouped, cut_off):
    s = grouped.agg(
        APP=("NSID", "count"),
        SAT=("CLEAN_MARKS", lambda x: x.notna().sum()),
        Above_70=("CLEAN_MARKS", lambda x: (x >= 70).sum()),
        Above_CutOff=("CLEAN_MARKS", lambda x: (x >= cut_off).sum()),
    ).reset_index()
    s["P_70"] = np.where(
        s["SAT"] > 0, ((s["Above_70"] / s["SAT"]) * 100).round(2), 0.0)
    s["P_CutOff"] = np.where(
        s["SAT"] > 0, ((s["Above_CutOff"] / s["SAT"]) * 100).round(2), 0.0)
    s["P_70_Rank"] = s["P_70"].rank(method="min", ascending=False).astype(int)
    s["P_CutOff_Rank"] = s["P_CutOff"].rank(
        method="min", ascending=False).astype(int)
    return s


def exam_department_analysis(raw: pd.DataFrame, cut_off: float, exclude_schools):
    df = raw.rename(columns={
        "No": "Index", "ZONE": "ZONE", "Name of School": "SCHNAME",
        "Name of Students": "Name_of_Students", "Part-I": "Part_I",
        "Part - II": "Part_II", "Total": "TOTAL_MARKS"}).copy()

    if "SCHID" not in df.columns:
        df["SCHID"] = df["SCHNAME"]
    df["SCHNAME"] = df["SCHNAME"].astype(str).str.strip()
    exclude = [s.strip() for s in exclude_schools if s.strip()]
    if exclude:
        df = df[~df["SCHNAME"].isin(exclude)].copy()

    df["NSID"] = np.arange(1, len(df) + 1)
    df["TOTAL_MARKS"] = df["TOTAL_MARKS"].astype(str).str.strip()
    df["ZONE"] = df["ZONE"].astype(str).str.strip()
    df["CLEAN_MARKS"] = pd.to_numeric(df["TOTAL_MARKS"], errors="coerce")

    bins = [-1, 9, 19, 29, 39, 49, 59, 69, 79, 99, 130, 149, np.inf]
    labels = ["1-9", "10-19", "20-29", "30-39", "40-49", "50-59",
              "60-69", "70-79", "80-99", "100-130", "131-149", "150+"]
    df["MARK_RANGE"] = pd.cut(
        df["CLEAN_MARKS"], bins=bins, labels=labels, include_lowest=True)
    df["MARK_RANGE"] = df["MARK_RANGE"].astype(str).replace("nan", "Absent")
    order = labels + ["Absent"]

    school_summary = _perf_table(df.groupby(["SCHID", "SCHNAME"]), cut_off)
    zone_summary = _perf_table(df.groupby(["ZONE"]), cut_off)
    school_dist = (pd.crosstab([df["SCHID"], df["SCHNAME"]], df["MARK_RANGE"])
                   .reindex(columns=order, fill_value=0).reset_index())
    zone_dist = (pd.crosstab(df["ZONE"], df["MARK_RANGE"])
                 .reindex(columns=order, fill_value=0).reset_index())
    return dict(school_summary=school_summary, zone_summary=zone_summary,
                school_dist=school_dist, zone_dist=zone_dist)


# =====================================================================
# Excel export
# =====================================================================
def build_excel(df, school_summary, target_students, adv, exam) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Analyzed_Students", index=False)
        # index kept so the school name is not lost
        school_summary.to_excel(w, sheet_name="School_Performance_Metrics")
        target_students.to_excel(
            w, sheet_name="Target_Remedial_Students", index=False)
        if adv is not None:
            adv["analysis_df"].to_excel(
                w, sheet_name="Raw_Meaningful_Data", index=False)
            if len(adv["freq_df"]) > 0:
                adv["freq_df"].to_excel(
                    w, sheet_name="Frequency_Distribution", index=False)
            adv["stats_summary"].to_excel(
                w, sheet_name="Summary_Stats", index=False)
            adv["score_table"].to_excel(
                w, sheet_name="Z_T_Scores", index=False)
            adv["corr_matrix"].to_excel(w, sheet_name="Correlation")
            adv["prob_df"].to_excel(
                w, sheet_name="Normal_Probabilities", index=False)
            adv["normality"].to_excel(
                w, sheet_name="Normality_Tests", index=False)
        exam["school_summary"].to_excel(
            w, sheet_name="School_Performance", index=False)
        exam["zone_summary"].to_excel(
            w, sheet_name="Zone_Performance", index=False)
        exam["school_dist"].to_excel(
            w, sheet_name="School_Score_Distributions", index=False)
        exam["zone_dist"].to_excel(
            w, sheet_name="Zone_Score_Distributions", index=False)
    return buf.getvalue()


# =====================================================================
# Figures (each returns a matplotlib Figure)
# =====================================================================
def fig_complete_analysis(c):
    a, freq_df, freq_table, bins = c["a"], c["freq_df"], c["freq_table"], c["bins"]
    cw, mean_val, median_val = c["class_width"], c["mean_val"], c["median_val"]
    std, dmin, dmax, n = c["std_sample"], c["data_min"], c["data_max"], c["n"]

    fig = plt.figure(figsize=(16, 22))
    ax1 = plt.subplot(5, 2, 1)
    ax1.hist(a["Score"], bins=bins, color="steelblue",
             edgecolor="black", alpha=0.8)
    ax1.axvline(mean_val, color="red", linestyle="--",
                linewidth=2, label=f"Mean={mean_val:.1f}")
    ax1.axvline(median_val, color="green", linestyle="--",
                linewidth=2, label=f"Median={median_val}")
    ax1.set(title="Histogram of Scores", xlabel="Score", ylabel="Frequency")
    ax1.legend()

    ax2 = plt.subplot(5, 2, 2)
    x_poly = y_poly = None
    if len(freq_df) > 0:
        mid = freq_df["Midpoint (x)"].values
        fr = freq_df["Frequency (f)"].values
        x_poly = np.concatenate([[mid[0] - cw], mid, [mid[-1] + cw]])
        y_poly = np.concatenate([[0], fr, [0]])
        ax2.plot(x_poly, y_poly, marker="o", color="darkorange", linewidth=2)
        ax2.fill_between(x_poly, y_poly, alpha=0.25, color="orange")
    ax2.set(title="Frequency Polygon",
            xlabel="Score Midpoint", ylabel="Frequency")

    ax3 = plt.subplot(5, 2, 3)
    if x_poly is not None and len(x_poly) >= 4:
        xs = np.linspace(x_poly.min(), x_poly.max(), 300)
        ys = make_interp_spline(x_poly, y_poly, k=3)(xs)
        ax3.plot(xs, ys, color="purple", linewidth=2)
        ax3.fill_between(xs, ys, alpha=0.2, color="purple")
    ax3.set(title="Frequency Curve (Smoothed)",
            xlabel="Score", ylabel="Frequency Density")

    ax4 = plt.subplot(5, 2, 4)
    if len(freq_df) > 0:
        ub = [iv.right for iv in freq_table.index]
        ax4.plot([dmin] + ub, [0] + list(freq_df["Cumulative f"].values),
                 marker="s", color="teal", linewidth=2)
    ax4.set(title="Cumulative Frequency (Ogive)",
            xlabel="Score", ylabel="Cumulative Frequency")

    ax5 = plt.subplot(5, 2, 5)
    xa = np.linspace(dmin - 5, dmax + 5, 500)
    pdf = norm.pdf(xa, mean_val, std if std > 0 else 1) * \
        n * (cw if cw > 0 else 1)
    ax5.hist(a["Score"], bins=bins, color="lightgray",
             edgecolor="black", alpha=0.7, label="Histogram")
    ax5.plot(xa, pdf, "r-", linewidth=2, label="Normal Curve")
    ax5.set(title="Histogram + Fitted Normal Distribution",
            xlabel="Score", ylabel="Frequency")
    ax5.legend()

    ax6 = plt.subplot(5, 2, 6)
    ax6.boxplot(a["Score"], orientation="horizontal", patch_artist=True,
                boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red", linewidth=2))
    ax6.set(title="Box Plot of Scores", xlabel="Score")
    ax6.set_yticks([])

    ax7 = plt.subplot(5, 2, 7)
    ax7.hist(a["Z_Score"], bins=10, color="mediumseagreen",
             edgecolor="black", alpha=0.8)
    ax7.axvline(0, color="red", linestyle="--",
                linewidth=2, label="Mean (z=0)")
    ax7.set(title="Z-Score Distribution", xlabel="Z-Score", ylabel="Frequency")
    ax7.legend()

    ax8 = plt.subplot(5, 2, 8)
    ax8.hist(a["T_Score"], bins=10, color="coral",
             edgecolor="black", alpha=0.8)
    ax8.axvline(50, color="red", linestyle="--",
                linewidth=2, label="Mean (T=50)")
    ax8.set(title="T-Score Distribution", xlabel="T-Score", ylabel="Frequency")
    ax8.legend()

    ax9 = plt.subplot(5, 2, 9)
    sns.heatmap(c["corr_matrix"], annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, ax=ax9, vmin=-1, vmax=1)
    ax9.set_title("Correlation Heatmap")

    ax10 = plt.subplot(5, 2, 10)
    ax10.scatter(a["Score"], a["T_Score"], color="darkblue",
                 alpha=0.7, edgecolor="black")
    ax10.set(
        title=f"Score vs T-Score  (r = {c['r']:.3f})", xlabel="Raw Score", ylabel="T-Score")
    ax10.grid(True)
    fig.tight_layout()
    return fig


def _qq_data(a):
    (osm, osr), (slope, intercept, r_qq) = probplot(
        a["Score"], dist="norm", plot=None)
    return osm, osr, slope, intercept, r_qq, slope * osm + intercept


def fig_qq(c):
    a, n, mean_val = c["a"], c["n"], c["mean_val"]
    osm, osr, slope, intercept, r_qq, line = _qq_data(a)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(osm, osr, color="steelblue", s=60, edgecolor="black", alpha=0.8,
               label="Observed Scores", zorder=3)
    ax.plot(osm, line, color="red", linewidth=2, linestyle="--",
            label=f"Normal Reference (R² = {r_qq**2:.4f})", zorder=2)
    se = slope / np.sqrt(n)
    band = 1.96 * se * (1 + osm ** 2) ** 0.5
    ax.fill_between(osm, line - band, line + band, color="gray", alpha=0.2,
                    label="95% Confidence Band")
    ranks = a["Score"].rank()
    for idx, row in a[np.abs(a["Z_Score"]) > 2].iterrows():
        theo_q = norm.ppf((ranks[idx] - 0.5) / n)
        ax.annotate(str(row["Student"]), (theo_q, row["Score"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    ax.set_title("Q-Q Plot: Student Scores vs. Normal Distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Theoretical Quantiles (Standard Normal)")
    ax.set_ylabel("Observed Sample Quantiles (Scores)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(mean_val, color="green", linestyle=":", alpha=0.6)
    ax.axvline(0, color="green", linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


def fig_normality_panel(c):
    a, bins, mean_val, std = c["a"], c["bins"], c["mean_val"], c["std_sample"]
    osm, osr, _, _, r_qq, line = _qq_data(a)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    ax.hist(a["Score"], bins=bins, density=True,
            color="lightsteelblue", edgecolor="black", alpha=0.8)
    xs = np.linspace(c["data_min"] - 5, c["data_max"] + 5, 400)
    ax.plot(xs, norm.pdf(xs, mean_val, std if std > 0 else 1),
            "r-", linewidth=2, label="Normal PDF")
    ax.set_title("A. Histogram vs Normal Curve", fontweight="bold")
    ax.set(xlabel="Score", ylabel="Density")
    ax.legend()
    ax = axes[1]
    ax.scatter(osm, osr, color="steelblue", s=40,
               edgecolor="black", alpha=0.85)
    ax.plot(osm, line, "r--", linewidth=2)
    ax.set_title(f"B. Q-Q Plot (R² = {r_qq**2:.3f})", fontweight="bold")
    ax.set(xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")
    ax = axes[2]
    ax.scatter(osm, osr - line, color="darkorange",
               s=40, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_title("C. Detrended Q-Q (Residuals)", fontweight="bold")
    ax.set(xlabel="Theoretical Quantiles", ylabel="Deviation from Normal")
    fig.tight_layout()
    return fig


def _with_bracket(df):
    df = df.copy()
    df["Score_Bracket"] = pd.cut(df["Average"], bins=[0, 35, 50, 70, 100],
                                 labels=["<35", "35-50", "50-70", ">70"], include_lowest=True)
    return df


def fig_cat1(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CATEGORY 1: DISTRIBUTION ANALYSIS",
                 fontsize=16, fontweight="bold", y=0.98)
    ax = axes[0, 0]
    ax.hist(df["Average"], bins=20, color="steelblue", edgecolor="black", alpha=0.7,
            density=True, label="Histogram")
    df["Average"].plot(kind="kde", ax=ax, color="red",
                       linewidth=2, label="KDE Curve")
    ax.axvline(df["Average"].mean(), color="green", linestyle="--", linewidth=2,
               label=f'Mean={df["Average"].mean():.1f}')
    ax.set(title="1.1 Histogram + KDE Overlay",
           xlabel="Average Score", ylabel="Density")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.boxplot(df["Average"], orientation="vertical", patch_artist=True,
               boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red", linewidth=2))
    ax.set(title="1.2 Overall Box Plot", ylabel="Average Score")
    ax.set_xticks([1])
    ax.set_xticklabels(["All Students"])

    ax = axes[0, 2]
    top = df["Name_of_School"].value_counts().head(10).index
    sns.violinplot(data=df[df["Name_of_School"].isin(top)], x="Name_of_School", y="Average",
                   ax=ax, hue="Name_of_School", palette="Set2", inner="quartile", legend=False)
    ax.set(title="1.3 Violin Plot (Top 10 Schools)",
           xlabel="School", ylabel="Average Score")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 0]
    sample = df.sample(min(200, len(df)), random_state=42)
    sns.stripplot(data=sample, x="P1_above_35", y="Average", ax=ax, jitter=True,
                  alpha=0.6, color="purple", size=5)
    ax.set(title="1.4 Strip Plot (Part-I Status)",
           xlabel="Part-I Status", ylabel="Average")

    ax = axes[1, 1]
    ax.hist(df["Average"], bins=20, color="lightgreen",
            edgecolor="black", alpha=0.6)
    for s in df["Average"]:
        ax.axvline(s, color="darkgreen", alpha=0.1, linewidth=0.5)
    ax.set(title="1.5 Histogram + Rug Plot",
           xlabel="Average Score", ylabel="Frequency")

    ax = axes[1, 2]
    for status, color in [("Q", "green"), ("NQ", "red")]:
        sub = df[df["above_70"] == status]["Average"]
        if len(sub) > 1:
            sub.plot(kind="kde", ax=ax,
                     label=f"{status} (Qualified)", color=color, linewidth=2)
    ax.set(title="1.6 Density by Qualification",
           xlabel="Average Score", ylabel="Density")
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_cat2(df):
    df = _with_bracket(df)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CATEGORY 2: COMPARISON ANALYSIS",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    sa = df.groupby("Name_of_School")[
        "Average"].mean().nlargest(10).sort_values()
    ax.barh(sa.index, sa.values, color=plt.cm.viridis(
        np.linspace(0.3, 1, len(sa))), edgecolor="black")
    ax.set(title="2.1 Top 10 Schools (Avg Score)", xlabel="Average Score")

    ax = axes[0, 1]
    ss = (df.assign(_p=(df["Average"] > 35).astype(int), _f=(df["Average"] <= 35).astype(int))
          .groupby("Name_of_School")[["_p", "_f"]].sum()
          .rename(columns={"_p": "Passed", "_f": "Failed"}).nlargest(8, "Passed"))
    ss.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax.set(title="2.2 Pass/Fail by School (Top 8)",
           xlabel="School", ylabel="Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Status")

    ax = axes[1, 0]
    bd = df.groupby(["Name_of_School", "Score_Bracket"],
                    observed=False).size().unstack(fill_value=0)
    bd.loc[bd.sum(axis=1).nlargest(8).index].plot(kind="bar", stacked=True, ax=ax,
                                                  colormap="RdYlGn", edgecolor="black")
    ax.set(title="2.3 Stacked Bar (Score Brackets)",
           xlabel="School", ylabel="Students")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Score Range", bbox_to_anchor=(1.05, 1))

    ax = axes[1, 1]
    sp = df.groupby("Name_of_School")[
        ["Part_I", "Part_II"]].mean().nlargest(8, "Part_I")
    x = np.arange(len(sp))
    w = 0.35
    ax.bar(x - w / 2, sp["Part_I"], w, label="Part-I",
           color="#3498db", edgecolor="black")
    ax.bar(x + w / 2, sp["Part_II"], w, label="Part-II",
           color="#e67e22", edgecolor="black")
    ax.set(title="2.4 Part-I vs Part-II by School",
           xlabel="School", ylabel="Average Marks")
    ax.set_xticks(x)
    ax.set_xticklabels(sp.index, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_pairplot(df):
    sample = df[["Part_I", "Part_II", "Average", "above_70"]].sample(
        min(300, len(df)), random_state=42)
    g = sns.pairplot(sample, hue="above_70", palette="Set1", diag_kind="kde",
                     plot_kws={"alpha": 0.6, "edgecolor": "black", "s": 30})
    g.figure.suptitle("CATEGORY 3.4: PAIR PLOT (Multi-Variable Relationships)",
                      y=1.02, fontsize=14, fontweight="bold")
    return g.figure


def fig_cat3(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CATEGORY 3: RELATIONSHIP ANALYSIS",
                 fontsize=16, fontweight="bold", y=0.98)
    ax = axes[0, 0]
    sc = ax.scatter(df["Part_I"], df["Part_II"], c=df["Average"], cmap="viridis",
                    alpha=0.6, edgecolors="black", s=40)
    ax.plot([0, 100], [0, 100], "r--", linewidth=1.5,
            label="Perfect Balance Line")
    ax.set(title="3.1 Part-I vs Part-II", xlabel="Part-I", ylabel="Part-II")
    plt.colorbar(sc, ax=ax, label="Average")
    ax.legend()

    ax = axes[0, 1]
    sb = (df.groupby("Name_of_School")
          .agg(Avg=("Average", "mean"), Count=("Average", "count"),
               Pass_Rate=("Average", lambda x: (x > 35).mean() * 100))
          .reset_index().nlargest(15, "Avg"))
    b = ax.scatter(sb["Avg"], sb["Pass_Rate"], s=sb["Count"] * 10, c=sb["Pass_Rate"],
                   cmap="plasma", alpha=0.7, edgecolors="black")
    ax.set(title="3.2 Bubble Chart", xlabel="Avg Score", ylabel="Pass Rate %")
    plt.colorbar(b, ax=ax)

    ax = axes[1, 0]
    corr = df[["Part_I", "Part_II", "Total", "Average",
               "Z_Score", "T_Score", "Diff_from_35"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("3.3 Correlation Heatmap")

    ax = axes[1, 1]
    hb = ax.hexbin(df["Part_I"], df["Part_II"],
                   gridsize=25, cmap="YlOrRd", mincnt=1)
    ax.set(title="3.4 Hexbin Plot (Density)",
           xlabel="Part-I", ylabel="Part-II")
    plt.colorbar(hb, ax=ax, label="Count")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_cat4(df, target_pct):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CATEGORY 4: RANKING ANALYSIS",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    ts = df.nlargest(15, "Average")[
        ["Name_of_Students", "Average"]].sort_values("Average")
    ax.barh(ts["Name_of_Students"], ts["Average"],
            color=plt.cm.RdYlGn(np.linspace(0.3, 1, len(ts))), edgecolor="black")
    ax.set(title="4.1 Top 15 Students Ranking", xlabel="Average Score")

    ax = axes[0, 1]
    sr = df.groupby("Name_of_School")[
        "Average"].mean().nlargest(12).sort_values()
    y = np.arange(len(sr))
    ax.hlines(y=y, xmin=0, xmax=sr.values,
              color="gray", alpha=0.5, linewidth=2)
    ax.scatter(sr.values, y, c=sr.values, cmap="cool",
               s=100, edgecolors="black", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sr.index)
    ax.set(title="4.2 Lollipop Chart (School Ranking)", xlabel="Average Score")

    axes[1, 0].remove()
    ax = fig.add_subplot(2, 2, 3, projection="polar")
    top5 = (df.groupby("Name_of_School")
            .agg(Avg=("Average", "mean"),
                 Pass_Rate=("Average", lambda x: (x > 35).mean() * 100),
                 Count=("Average", "count"),
                 Above_70=("above_70", lambda x: (x == "Q").mean() * 100),
                 Std=("Average", "std"))
            .nlargest(5, "Avg").fillna(0))
    if not top5.empty:
        top5["Count_Norm"] = (top5["Count"] / top5["Count"].max()) * 100
        top5["Std_Norm"] = (100 - (top5["Std"] / top5["Std"].max()) * 100
                            if top5["Std"].max() > 0 else 100)
        cats = ["Avg Score", "Pass Rate %",
                "Student Count", "Above 70 %", "Consistency"]
        N = len(cats)
        ang = [i / float(N) * 2 * np.pi for i in range(N)]
        ang += ang[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(cats, size=9)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"],
                           color="grey", size=7)
        ax.set_ylim(0, 100)
        for school, r in top5.iterrows():
            v = [r["Avg"], r["Pass_Rate"], r["Count_Norm"],
                 r["Above_70"], r["Std_Norm"]]
            v += v[:1]
            ax.plot(ang, v, linewidth=2, label=str(school)[:20])
            ax.fill(ang, v, alpha=0.1)
        ax.set_title("4.3 Radar Chart (Top 5 Schools)", y=1.10, fontsize=11)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)
    else:
        ax.set_title("4.3 Radar Chart — insufficient data", y=1.10)
        ax.set_axis_off()

    ax = axes[1, 1]
    sc = (df.assign(_c=(df["Average"] > 35).astype(int))
          .groupby("Name_of_School")["_c"].mean().mul(100).to_frame("Current")
          .assign(Target=float(target_pct)).nlargest(10, "Current").sort_values("Current"))
    for i, (_, r) in enumerate(sc.iterrows()):
        ax.plot([r["Current"], r["Target"]], [i, i],
                "gray", linewidth=3, alpha=0.5)
        ax.scatter(r["Current"], i, color="#3498db", s=100, edgecolors="black", zorder=3,
                   label="Current" if i == 0 else "")
        ax.scatter(r["Target"], i, color="#e74c3c", s=100, marker="D", edgecolors="black",
                   zorder=3, label="Target" if i == 0 else "")
    ax.set_yticks(np.arange(len(sc)))
    ax.set_yticklabels(sc.index)
    ax.set(title="4.4 Dumbbell Plot (Current vs Target)", xlabel="Percentage (%)")
    ax.legend(loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_cat5(df):
    df = _with_bracket(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("CATEGORY 5: COMPOSITION ANALYSIS",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    p = int((df["Average"] > 35).sum())
    f = len(df) - p
    _, _, at = ax.pie([p, f], explode=(0.05, 0.05), labels=[f"Passed\n({p})", f"Failed\n({f})"],
                      colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", shadow=True, startangle=90)
    for t in at:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.set_title("5.1 Pass/Fail Distribution")

    ax = axes[0, 1]
    bc = df["Score_Bracket"].value_counts().sort_index()
    ax.pie(bc.values, labels=bc.index, colors=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"][:len(bc)],
           autopct="%1.1f%%", startangle=90, pctdistance=0.85,
           wedgeprops=dict(width=0.4, edgecolor="white"))
    ax.text(0, 0, f"Total\n{len(df)}", ha="center",
            va="center", fontsize=14, fontweight="bold")
    ax.set_title("5.2 Donut Chart (Score Brackets)")

    ax = axes[1, 0]
    cd = df.groupby("Name_of_School").size().nlargest(8)
    ax.bar(range(len(cd)), cd.values, color="steelblue", edgecolor="black")
    ax.fill_between(range(len(cd)), cd.values.cumsum(),
                    alpha=0.3, color="orange")
    ax.set(title="5.3 Bar + Cumulative Area",
           xlabel="School (Top 8)", ylabel="Students")
    ax.set_xticks(range(len(cd)))
    ax.set_xticklabels([str(s)[:15]
                       for s in cd.index], rotation=45, ha="right")

    ax = axes[1, 1]
    zs = df.groupby(["Zone", "Name_of_School"]
                    ).size().reset_index(name="Count")
    if not zs.empty:
        top_zone = zs.groupby("Zone")["Count"].sum().idxmax()
        zd = zs[zs["Zone"] == top_zone].nlargest(
            10, "Count").sort_values("Count")
        ax.barh(zd["Name_of_School"], zd["Count"],
                color="mediumpurple", edgecolor="black")
        ax.set(
            title=f"5.4 Treemap Style (Zone: {top_zone})", xlabel="Student Count")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_cat6(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CATEGORY 6: STATISTICAL & ADVANCED ANALYSIS",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    se = df.groupby("Name_of_School")["Average"].agg(
        ["mean", "std", "count"]).nlargest(10, "mean")
    se["se"] = se["std"] / np.sqrt(se["count"])
    x = np.arange(len(se))
    ax.errorbar(x, se["mean"], yerr=se["se"], fmt="o", color="darkblue", ecolor="red",
                capsize=5, markersize=8, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s)[:15]
                       for s in se.index], rotation=45, ha="right")
    ax.set(title="6.1 Error Bars (Mean ± SE)", ylabel="Average Score")

    ax = axes[0, 1]
    sw = df.sample(min(500, len(df)), random_state=42)
    sns.swarmplot(data=sw, x="above_70", y="Average", ax=ax, hue="above_70",
                  palette={"Q": "green", "NQ": "red"}, size=4, alpha=0.7, legend=False)
    ax.set(title="6.2 Swarm Plot (Qualification Status)",
           xlabel="Status", ylabel="Average")

    ax = axes[1, 0]
    ss = np.sort(df["Average"])
    ax.step(ss, np.arange(1, len(ss) + 1) / len(ss),
            color="purple", linewidth=2, where="post")
    ax.axvline(35, color="red", linestyle="--",
               linewidth=2, label="Pass Threshold (35)")
    ax.axvline(df["Average"].mean(), color="green", linestyle="--", linewidth=2,
               label=f'Mean ({df["Average"].mean():.1f})')
    ax.set(title="6.3 ECDF Plot", xlabel="Average Score",
           ylabel="Cumulative Probability")
    ax.legend()

    ax = axes[1, 1]
    ci = df.groupby("Name_of_School")["Average"].agg(
        ["mean", "count", "std"]).nlargest(10, "mean")
    ci["ci"] = 1.96 * ci["std"] / np.sqrt(ci["count"])
    y = np.arange(len(ci))
    ax.barh(y, ci["mean"], xerr=ci["ci"], color="teal",
            edgecolor="black", alpha=0.7, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([str(s)[:15] for s in ci.index])
    ax.set(title="6.4 95% Confidence Intervals", xlabel="Average Score")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def build_all_graphs(df, ctx, target_pct):
    """Returns {filename: (caption, png_bytes)} — a failure in one graph never stops the rest."""
    jobs = []
    if ctx is not None:
        jobs += [("complete_analysis.png", "Complete statistical analysis", lambda: fig_complete_analysis(ctx)),
                 ("qq_plot.png", "Q-Q plot", lambda: fig_qq(ctx)),
                 ("normality_panel.png", "Normality panel", lambda: fig_normality_panel(ctx))]
    jobs += [
        ("01_Distribution_Analysis.png",
         "Category 1 – Distribution", lambda: fig_cat1(df)),
        ("02_Comparison_Analysis.png", "Category 2 – Comparison", lambda: fig_cat2(df)),
        ("03_Relationship_Analysis.png",
         "Category 3 – Relationship", lambda: fig_cat3(df)),
        ("03d_Pair_Plot.png", "Category 3 – Pair plot", lambda: fig_pairplot(df)),
        ("04_Ranking_Analysis.png", "Category 4 – Ranking",
         lambda: fig_cat4(df, target_pct)),
        ("05_Composition_Analysis.png",
         "Category 5 – Composition", lambda: fig_cat5(df)),
        ("06_Statistical_Analysis.png",
         "Category 6 – Statistical", lambda: fig_cat6(df)),
    ]
    out, errors = {}, {}
    for fname, caption, fn in jobs:
        try:
            out[fname] = (caption, fig_to_png(fn()))
        except Exception as e:  # noqa: BLE001
            plt.close("all")
            errors[fname] = str(e)
    return out, errors


# =====================================================================
# Cached wrappers (keep the UI snappy when widgets change)
# =====================================================================
@st.cache_data(show_spinner=False)
def cached_marks(raw, target_pct):
    df, false_rec, needed, school_summary, targets = clean_and_analyze(
        raw, target_pct)
    adv = advanced_statistics(df)
    return df, false_rec, needed, school_summary, targets, adv


@st.cache_data(show_spinner=False)
def cached_exam(raw, cutoff, exclude):
    return exam_department_analysis(raw, cutoff, list(exclude))


@st.cache_data(show_spinner=False)
def cached_graphs(df, _ctx, ctx_key, target_pct):
    return build_all_graphs(df, _ctx, target_pct)


@st.cache_data(show_spinner=False)
def cached_excel(df, school_summary, targets, _adv, adv_key, _exam, exam_key):
    return build_excel(df, school_summary, targets, _adv, _exam)


# =====================================================================
# UI
# =====================================================================
def main():
    st.set_page_config(page_title="Zone Marks Analysis",
                       page_icon="📊", layout="wide")
    st.title("📊 Zone Marks Analysis")
    st.caption("Upload the school marks workbook to get pass-rate analysis, remedial targets, "
               "descriptive statistics, normality tests, zone/school summaries and graphs.")

    with st.sidebar:
        st.header("Inputs")
        up = st.file_uploader("Marks workbook (.xlsx)", type=["xlsx", "xls"])
        sheet = None
        target_pct = st.number_input(
            "Target pass percentage (%)", 0.0, 100.0, 75.0, 1.0)
        cutoff = st.number_input(
            "Cut-off marks (Total)", 0.0, 300.0, 35.0, 1.0)
        excl_text = st.text_area(
            "Schools to exclude from the Exam Department analysis (one per line)",
            value="T/SINHALA MAHA VIDYALAYA,", height=90)
        st.caption("Expected columns: No, ZONE, Name of School, Name of Students, "
                   "Part-I, Part - II, Total")

    if up is None:
        st.info("⬅️ Upload an Excel file in the sidebar to begin.")
        return

    try:
        xl = pd.ExcelFile(up)
        if len(xl.sheet_names) > 1:
            with st.sidebar:
                sheet = st.selectbox("Sheet", xl.sheet_names)
        raw = xl.parse(sheet or xl.sheet_names[0])
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")
        return

    miss = missing_columns(raw)
    if miss:
        st.error(f"Missing required column(s): {', '.join(miss)}")
        st.write("Columns found:", list(raw.columns))
        return

    exclude = tuple(l for l in excl_text.splitlines() if l.strip())
    try:
        with st.spinner("Analysing…"):
            df, false_rec, needed, school_summary, targets, adv = cached_marks(
                raw, target_pct)
            exam = cached_exam(raw, cutoff, exclude)
    except Exception as e:  # noqa: BLE001
        st.error(f"Analysis failed: {e}")
        return

    passed = int((df["Average"] > 35).sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Passed (Avg > 35)", f"{passed:,}",
              f"{passed / len(df) * 100:.1f}%")
    c3.metric("Mean score", f"{df['Average'].mean():.2f}")
    c4.metric("Std dev (σ)", f"{df['Average'].std(ddof=0):.2f}")
    c5.metric(f"Passes needed for {target_pct:g}%",
              f"{max(needed, 0):,}" if needed > 0 else "Target met")

    tabs = st.tabs(["Overview", "Schools & targets", "Remedial students", "Statistics",
                    "Exam department", "Graphs", "Download"])

    with tabs[0]:
        st.subheader("Data integrity")
        if false_rec.empty:
            st.success("All Total values match Part-I + Part-II.")
        else:
            st.warning(
                f"{len(false_rec)} record(s) where Total ≠ Part-I + Part-II")
            st.dataframe(false_rec[["Name_of_Students", "Part_I", "Part_II", "Total", "Check_Total"]],
                         use_container_width=True)
        p1 = df.loc[df["Part_I"] > 0, "Part_I"]
        p2 = df.loc[df["Part_II"] > 0, "Part_II"]
        a, b = st.columns(2)
        for col, name, s in ((a, "Part I", p1), (b, "Part II", p2)):
            with col:
                st.markdown(f"**{name} (excluding 0)**")
                if s.empty:
                    st.write("No non-zero scores")
                else:
                    st.write(
                        f"Min {s.min():g} · Max {s.max():g} · Range {s.max() - s.min():g}")
        st.subheader("Analysed students")
        st.dataframe(df, use_container_width=True, height=380)

    with tabs[1]:
        st.subheader("School summary")
        st.caption(f"Target {target_pct:g}% — sorted by passes still needed")
        st.dataframe(school_summary.sort_values("Needed_Passes", ascending=False)
                     .style.format({"Current_%": "{:.1f}"}), use_container_width=True)
        top = school_summary.sort_values(
            "Needed_Passes", ascending=False).head(15)
        st.bar_chart(top["Needed_Passes"])

    with tabs[2]:
        st.subheader(f"Remedial students (Average < 35): {len(targets)}")
        sch = st.multiselect("Filter by school", sorted(
            targets["Name_of_School"].dropna().unique()))
        view = targets[targets["Name_of_School"].isin(sch)] if sch else targets
        st.dataframe(view[["Name_of_Students", "Name_of_School", "Zone", "Part_I", "Part_II", "Average"]],
                     use_container_width=True, height=450)

    with tabs[3]:
        if adv is None:
            st.warning("No valid scores for advanced analysis.")
        else:
            l, r = st.columns(2)
            with l:
                st.subheader("Summary statistics")
                st.dataframe(adv["stats_summary"].astype({"Value": str}), use_container_width=True,
                             hide_index=True)
            with r:
                st.subheader("Normality tests")
                st.dataframe(adv["normality"].astype(str),
                             use_container_width=True, hide_index=True)
                st.subheader("Normal probabilities")
                st.dataframe(adv["prob_df"],
                             use_container_width=True, hide_index=True)
            st.subheader("Frequency distribution")
            st.dataframe(adv["freq_df"],
                         use_container_width=True, hide_index=True)
            st.subheader("Z / T scores & percentiles")
            st.dataframe(adv["score_table"],
                         use_container_width=True, height=350)
            st.subheader("Correlation")
            st.dataframe(adv["corr_matrix"].style.format(
                "{:.3f}"), use_container_width=True)

    with tabs[4]:
        st.subheader("Zone performance")
        st.dataframe(exam["zone_summary"],
                     use_container_width=True, hide_index=True)
        st.subheader("School performance")
        st.dataframe(exam["school_summary"],
                     use_container_width=True, hide_index=True, height=350)
        st.subheader("Score distribution by zone")
        st.dataframe(exam["zone_dist"],
                     use_container_width=True, hide_index=True)
        st.subheader("Score distribution by school")
        st.dataframe(exam["school_dist"],
                     use_container_width=True, hide_index=True, height=350)

    ctx = adv["ctx"] if adv else None
    with tabs[5]:
        st.caption("Graph generation takes a few seconds; results are cached.")
        if st.button("Generate graphs", type="primary") or st.session_state.get("graphs_on"):
            st.session_state["graphs_on"] = True
            with st.spinner("Rendering graphs…"):
                graphs, errors = cached_graphs(
                    df, ctx, (len(df), target_pct), target_pct)
            for fname, (caption, png) in graphs.items():
                st.markdown(f"#### {caption}")
                st.image(png, use_container_width=True)
            for fname, msg in errors.items():
                st.warning(f"{fname} could not be generated: {msg}")

    with tabs[6]:
        xbytes = cached_excel(df, school_summary, targets,
                              adv, len(df), exam, cutoff)
        st.download_button("⬇️ Download consolidated Excel report", xbytes,
                           file_name="Combined_Zone_Marks_Analysis_Report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")
        if st.session_state.get("graphs_on"):
            graphs, _ = cached_graphs(
                df, ctx, (len(df), target_pct), target_pct)
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                for fname, (_, png) in graphs.items():
                    z.writestr(fname, png)
            st.download_button("⬇️ Download all graphs (ZIP)", zbuf.getvalue(),
                               file_name="Graphs_Output.zip", mime="application/zip")
        else:
            st.caption(
                "Generate the graphs (Graphs tab) to enable the graph ZIP download.")


if __name__ == "__main__":
    main()
