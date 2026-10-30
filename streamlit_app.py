"""EduMetrics — redesigned analytics dashboard for G05 Exam B."""

import html
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import stats


# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EduMetrics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════
# PLOTLY GLOBAL THEME
# ════════════════════════════════════════════════════════════
PALETTE = ["#6D28D9", "#0891B2", "#059669",
           "#D97706", "#DC2626", "#7C3AED", "#DB2777"]

pio.templates["edu"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif",
                  color="#0B0B0F", size=12.5),
        colorway=PALETTE,
        title=dict(font=dict(size=14, color="#0B0B0F", family="Inter"),
                   x=0, xanchor="left", pad=dict(b=16)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#F0EDE7", linecolor="#E8E5DF",
                   zerolinecolor="#F0EDE7",
                   tickfont=dict(size=11.5, color="#64646E")),
        yaxis=dict(gridcolor="#F0EDE7", linecolor="#E8E5DF",
                   zerolinecolor="#F0EDE7",
                   tickfont=dict(size=11.5, color="#64646E")),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E8E5DF",
                        font=dict(size=12, family="Inter", color="#0B0B0F")),
        margin=dict(l=40, r=20, t=48, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11.5),
                    orientation="h", y=-0.15, x=0),
        coloraxis=dict(colorbar=dict(tickfont=dict(size=10))),
    )
)
pio.templates.default = "edu"


# ════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500;600&display=swap');

:root {
    --ink:          #0B0B0F;
    --ink-2:        #1C1C24;
    --muted:        #64646E;
    --faint:        #A1A1AA;
    --line:         #E8E5DF;
    --line-soft:    #F0EDE7;
    --canvas:       #FBFAF7;
    --surface:      #FFFFFF;
    --surface-2:    #F5F3EE;
    --surface-3:    #EDEAE3;
    --accent:       #6D28D9;
    --accent-2:     #8B5CF6;
    --accent-soft:  #F2ECFF;
    --success:      #047857;
    --success-soft: #ECFDF5;
    --danger:       #B91C1C;
    --danger-soft:  #FEE2E2;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}
.stApp { background-color: var(--canvas); }
h1, h2, h3, h4, h5 { color: var(--ink); font-weight: 700; letter-spacing: -0.02em; }

/* Chrome cleanup */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0B0F 0%, #1C1C24 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
    min-width: 300px !important;
    max-width: 300px !important;
}
section[data-testid="stSidebar"] * { color: #E4E4E7; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); margin: 20px 0; }
section[data-testid="stSidebar"] label { color: #A1A1AA !important; font-size: 12px !important; font-weight: 500; }

.brand { display: flex; align-items: center; gap: 12px; padding: 8px 0 24px; }
.brand-mark {
    width: 40px; height: 40px; border-radius: 11px;
    background: linear-gradient(135deg, #6D28D9 0%, #8B5CF6 100%);
    color: #FFFFFF; display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 700; flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(109,40,217,0.35);
}
.brand-name { font-weight: 700; font-size: 15px; color: #FFFFFF;
              letter-spacing: -0.01em; line-height: 1.2; }
.brand-sub  { font-size: 10.5px; color: #71717A; letter-spacing: 0.14em;
              text-transform: uppercase; margin-top: 3px; }

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 12px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #E4E4E7 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg { fill: #A1A1AA !important; }

.side-status { padding: 12px 14px; border-radius: 10px; font-size: 12.5px;
               font-weight: 500; margin-top: 12px; }
.side-status.ok { background: rgba(4,120,87,0.12); color: #6EE7B7 !important;
                  border: 1px solid rgba(4,120,87,0.3); }
.side-status.err { background: rgba(185,28,28,0.12); color: #FCA5A5 !important;
                   border: 1px solid rgba(185,28,28,0.3); }

/* ── Nav (horizontal radio → pill tabs) ──────────────── */
[data-testid="stRadio"] > div[role="radiogroup"] {
    flex-direction: row !important;
    gap: 2px !important;
    background: var(--surface);
    padding: 5px;
    border-radius: 14px;
    border: 1px solid var(--line);
    width: fit-content;
    box-shadow: 0 1px 2px rgba(11,11,15,0.03);
}
[data-testid="stRadio"] label {
    padding: 9px 18px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    color: var(--muted) !important;
    cursor: pointer;
    transition: all .18s ease;
    margin: 0 !important;
}
[data-testid="stRadio"] label:hover { color: var(--ink) !important; background: var(--surface-2) !important; }
[data-testid="stRadio"] label:has(input:checked) {
    background: var(--ink) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(11,11,15,0.15);
}
[data-testid="stRadio"] label:has(input:checked):hover { background: var(--ink-2) !important; }
[data-testid="stRadio"] label > div:first-child { display: none !important; }

/* ── Pills (zone filter) ─────────────────────────────── */
[data-testid="stPills"] button {
    border-radius: 999px !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    color: var(--muted) !important;
    transition: all .18s ease;
}
[data-testid="stPills"] button:hover { border-color: var(--faint) !important; color: var(--ink) !important; }
[data-testid="stPills"] button[kind="pillsActive"],
[data-testid="stPills"] button[aria-checked="true"] {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Selectbox ───────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border-color: var(--line) !important;
    background: var(--surface) !important;
    font-size: 13.5px !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--ink) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(11,11,15,0.08);
}

/* ── Page head ───────────────────────────────────────── */
.page-head { padding: 4px 0 32px; }
.page-eyebrow {
    font-size: 11px; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--accent);
    display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
}
.page-eyebrow::before { content: ""; width: 6px; height: 6px;
                        background: var(--accent); border-radius: 50%; }
.page-head h1 {
    font-size: 46px !important; font-weight: 700 !important;
    letter-spacing: -0.04em !important; line-height: 1.05 !important;
    margin: 0 !important; color: var(--ink);
}
.page-head p { font-size: 15px; color: var(--muted); margin: 14px 0 0;
               max-width: 640px; line-height: 1.6; }

/* ── Section label ───────────────────────────────────── */
.section-label {
    font-size: 11.5px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--faint);
    margin: 40px 0 14px; display: flex; align-items: center; gap: 12px;
}
.section-label::after {
    content: ""; flex: 1; height: 1px; background: var(--line-soft);
}

/* ── KPI cards ───────────────────────────────────────── */
.kpi {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px 22px 20px;
    position: relative;
    overflow: hidden;
    transition: all .25s cubic-bezier(.4,0,.2,1);
    height: 100%;
}
.kpi::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--kpi-accent, var(--accent));
}
.kpi:hover { transform: translateY(-3px); box-shadow: 0 16px 40px rgba(11,11,15,0.08); }
.kpi-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
}
.kpi-value {
    font-family: 'Geist Mono', ui-monospace, monospace;
    font-size: 40px; font-weight: 600; letter-spacing: -0.03em;
    color: var(--ink); margin-top: 10px; line-height: 1;
}
.kpi-sub { font-size: 12px; color: var(--faint); margin-top: 10px; }
.kpi-delta {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 12px; font-weight: 600;
    padding: 3px 9px; border-radius: 999px; margin-top: 10px;
    background: var(--surface-2);
}
.kpi-delta.pos { color: var(--success); background: var(--success-soft); }
.kpi-delta.neg { color: var(--danger);  background: var(--danger-soft); }

/* ── Chart card ──────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 12px 8px 4px;
    box-shadow: 0 1px 2px rgba(11,11,15,0.03);
}

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
}

/* ── Leaderboard ─────────────────────────────────────── */
.lb { background: var(--surface); border: 1px solid var(--line);
      border-radius: 18px; padding: 8px 24px; }
.lb-row {
    display: grid; grid-template-columns: 36px 1fr 280px 72px;
    align-items: center; gap: 18px;
    padding: 12px 0; border-bottom: 1px solid var(--line-soft);
}
.lb-row:last-child { border-bottom: none; }
.lb-rank {
    font-family: 'Geist Mono', monospace; font-weight: 600;
    font-size: 13px; color: var(--faint); text-align: center;
}
.lb-rank.gold   { color: #B45309; }
.lb-rank.silver { color: #71717A; }
.lb-rank.bronze { color: #92400E; }
.lb-name { font-weight: 500; font-size: 13.5px; color: var(--ink);
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lb-bar { height: 6px; background: var(--surface-3);
          border-radius: 999px; overflow: hidden; }
.lb-fill { height: 100%; border-radius: 999px;
           background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.lb-val { font-family: 'Geist Mono', monospace; font-size: 12.5px;
          color: var(--ink); text-align: right; font-weight: 500; }

/* ── Empty state ─────────────────────────────────────── */
.empty {
    text-align: center; padding: 60px 20px; background: var(--surface);
    border: 1px dashed var(--line); border-radius: 18px; color: var(--muted);
}

/* ── Native tabs (stats page) ────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-weight: 600; font-size: 13.5px; color: var(--muted);
    padding: 10px 4px; background: transparent;
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════
def page_head(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-head">'
        f'<div class="page-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{html.escape(label)}</div>',
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", delta: str = "",
        accent: str = "var(--accent)", size: str = "") -> None:
    delta_html = ""
    if delta:
        cls = "neg" if delta.strip().startswith("-") else "pos"
        delta_html = f'<span class="kpi-delta {cls}">{html.escape(delta)}</span>'
    st.markdown(
        f'<div class="kpi {size}" style="--kpi-accent:{accent}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'{delta_html}'
        f'{f"<div class=kpi-sub>{html.escape(sub)}</div>" if sub else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# DATA PROCESSING  (pure, cache-safe)
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def process_uploaded_data(file_bytes: bytes) -> dict:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()

    rename_map = {
        "No": "Index", "ZONE": "Zone",
        "Name of School": "Name_of_School",
        "Name of Students": "Name_of_Students",
        "Part-I": "Part_I", "Part - I": "Part_I", "Part_I": "Part_I",
        "Part - II": "Part_II", "Part_II": "Part_II",
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["Part_I", "Part_II",
                "Name_of_School", "Name_of_Students", "Zone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # Numeric conversion + absent detection (before fillna)
    df["Part_I"] = pd.to_numeric(df["Part_I"],  errors="coerce")
    df["Part_II"] = pd.to_numeric(df["Part_II"], errors="coerce")
    df["Absent"] = df["Part_I"].isna() & df["Part_II"].isna()
    df["Part_I"] = df["Part_I"].fillna(0)
    df["Part_II"] = df["Part_II"].fillna(0)
    df["Total"] = df["Part_I"] + df["Part_II"]

    # Core derived metrics
    df["Average"] = df["Total"] / 2
    df["P1_above_35"] = np.where(df["Part_I"] >= 35, "QP1",  "NQP1")
    df["P2_above_35"] = np.where(df["Part_II"] >= 35, "QP2",  "NQP2")
    df["above_70"] = np.where((df["Part_I"] >= 35) &
                              (df["Part_II"] >= 35), "Q", "NQ")
    df["Diff_from_35"] = df["Average"] - 35

    mean_avg = df["Average"].mean()
    std_avg = df["Average"].std(ddof=0) or 1
    df["Z_Score"] = (df["Average"] - mean_avg) / std_avg
    df["T_Score"] = 50 + 10 * df["Z_Score"]

    # Remedial: below 35 AND not absent
    remedial = df[(df["Average"] < 35) & (~df["Absent"])].copy()

    # School metrics
    school_total = df.groupby("Name_of_School")["Average"].count()
    school_pass = df.assign(_p=(df["Average"] >= 35).astype(int)) \
        .groupby("Name_of_School")["_p"].sum()
    school_metrics = pd.DataFrame({
        "Total_Students": school_total,
        "Passed":         school_pass,
        "Current_%":      (school_pass / school_total) * 100,
    }).reset_index()
    school_metrics["Target_Passes"] = (
        school_metrics["Total_Students"] * 0.8).round().astype(int)
    school_metrics["Needed_Passes"] = (
        school_metrics["Target_Passes"] - school_metrics["Passed"]
    ).clip(lower=0)

    # Mark range + absent
    df["CLEAN_MARKS"] = pd.to_numeric(df["Total"], errors="coerce")
    df.loc[df["Absent"], "CLEAN_MARKS"] = np.nan

    bins = [-1, 9, 19, 29, 39, 49, 59, 69, 79, 99, 130, 149, np.inf]
    labels = ["1-9", "10-19", "20-29", "30-39", "40-49", "50-59",
              "60-69", "70-79", "80-99", "100-130", "131-149", "150+"]
    df["MARK_RANGE"] = pd.cut(df["CLEAN_MARKS"], bins=bins, labels=labels,
                              include_lowest=True).astype(str).replace("nan", "Absent")

    CUTOFF = 35.0

    def _perf_by(col: str) -> pd.DataFrame:
        out = df.groupby(col).agg(
            APP=("Index", "count"),
            SAT=("CLEAN_MARKS", lambda x: x.notna().sum()),
            Above_70=("CLEAN_MARKS", lambda x: (x >= 70).sum()),
            Above_CutOff=("CLEAN_MARKS", lambda x: (x >= CUTOFF).sum()),
        ).reset_index()
        out["P_70"] = np.where(
            out["SAT"] > 0,
            ((out["Above_70"] / out["SAT"]) * 100).round(2),
            0.0,
        )
        return out

    school_perf = _perf_by("Name_of_School")
    zone_perf = _perf_by("Zone")

    # Statistics summary
    raw = df["Average"].dropna()
    stats_summary = pd.DataFrame({
        "Statistic": ["N", "Mean", "Median", "Std Dev", "Min", "Max",
                      "Q1", "Q3", "Skewness", "Kurtosis"],
        "Value": [
            len(raw), round(raw.mean(), 3), round(raw.median(), 3),
            round(raw.std(), 3), raw.min(), raw.max(),
            round(raw.quantile(0.25), 3), round(raw.quantile(0.75), 3),
            round(raw.skew(), 3), round(raw.kurtosis(), 3),
        ],
    })

    # Normality tests
    normality = pd.DataFrame()
    if len(raw) >= 8:
        sw_stat, sw_p = stats.shapiro(raw)
        ks_stat, ks_p = stats.kstest(
            raw, stats.norm.cdf, args=(raw.mean(), raw.std()))
        try:
            dag_stat, dag_p = stats.normaltest(raw)
        except Exception:
            dag_stat, dag_p = np.nan, np.nan
        try:
            ad = stats.anderson(raw, dist="norm", method="interpolate")
            ad_stat, ad_p = ad.statistic, float(getattr(ad, "pvalue", np.nan))
        except Exception:
            ad = stats.anderson(raw, dist="norm")
            ad_stat, ad_p = ad.statistic, np.nan

        normality = pd.DataFrame({
            "Test":      ["Shapiro-Wilk", "Kolmogorov-Smirnov",
                          "D'Agostino-Pearson", "Anderson-Darling"],
            "Statistic": [round(sw_stat, 4), round(ks_stat, 4),
                          round(dag_stat, 4) if not np.isnan(
                              dag_stat) else "N/A",
                          round(ad_stat, 4)],
            "p-value":   [round(sw_p, 4), round(ks_p, 4),
                          round(dag_p, 4) if not np.isnan(dag_p) else "N/A",
                          round(ad_p, 4) if not np.isnan(ad_p) else "N/A"],
            "Conclusion": [
                "Normal" if sw_p > 0.05 else "Not Normal",
                "Normal" if ks_p > 0.05 else "Not Normal",
                "Normal" if (not np.isnan(dag_p) and dag_p >
                             0.05) else "Not Normal",
                "Normal" if (not np.isnan(ad_p) and ad_p >
                             0.05) else "Not Normal",
            ],
        })

    return {
        "students":       df,
        "school_metrics": school_metrics,
        "remedial":       remedial,
        "school_perf":    school_perf,
        "zone_perf":      zone_perf,
        "stats_summary":  stats_summary,
        "normality":      normality,
        "absent_count":   int(df["Absent"].sum()),
    }


# ════════════════════════════════════════════════════════════
# SIDEBAR — brand + upload only
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-mark">◆</div>'
        '<div>'
        '<div class="brand-name">EduMetrics</div>'
        '<div class="brand-sub">Analytics</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload raw Excel", type=["xlsx"],
                                label_visibility="collapsed")

    if uploaded is None:
        st.markdown(
            '<div style="color:#71717A;font-size:12.5px;line-height:1.6;padding:16px 0;">'
            'Drop a <strong>G05_ExamB_Analysis.xlsx</strong> file to begin.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        with st.spinner("Processing…"):
            data = process_uploaded_data(uploaded.getvalue())
        st.markdown('<div class="side-status ok">◆ Data loaded</div>',
                    unsafe_allow_html=True)
    except Exception as exc:
        st.markdown(f'<div class="side-status err">◆ {html.escape(str(exc))}</div>',
                    unsafe_allow_html=True)
        st.stop()

    st.divider()
    st.markdown(
        '<div style="font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;'
        'color:#52525B;font-weight:700;margin-bottom:6px">Source</div>'
        f'<div style="font-size:12.5px;color:#A1A1AA">{html.escape(uploaded.name)}</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# NAV + FILTER BAR
# ════════════════════════════════════════════════════════════
PAGES = ["Overview", "Schools & Zones", "Statistics", "Remedial"]

nav_col, _ = st.columns([1, 1])
with nav_col:
    page = st.radio("nav", PAGES, horizontal=True,
                    label_visibility="collapsed")

# Zone pills + school selectbox + export
zones = ["All"] + sorted(data["students"]["Zone"].dropna().unique().tolist())
fc1, fc2, fc3 = st.columns([1.2, 3, 1], gap="medium")

with fc1:
    selected_zone = st.pills("Zone", zones, default="All",
                             label_visibility="collapsed") or "All"

filtered = data["students"]
if selected_zone != "All":
    filtered = filtered[filtered["Zone"] == selected_zone]

schools = ["All"] + \
    sorted(filtered["Name_of_School"].dropna().unique().tolist())
with fc2:
    selected_school = st.selectbox("School", schools,
                                   label_visibility="collapsed")

if selected_school != "All":
    filtered = filtered[filtered["Name_of_School"] == selected_school]

with fc3:
    st.download_button(
        "↓ Export CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "edumetrics_filtered.csv",
        "text/csv",
        width="stretch",
    )

st.markdown("")  # breathing room

if filtered.empty:
    st.markdown('<div class="empty">No students match the current filters.</div>',
                unsafe_allow_html=True)
    st.stop()

# Convenience derived values used across pages
total = len(filtered)
passed = int((filtered["above_70"] == "Q").sum())
pass_rate = (passed / total * 100) if total else 0.0
avg = filtered["Average"].mean() if total else 0.0


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    page_head(
        "G05 · Exam B",
        "Executive Overview",
        f"Real-time performance across {total:,} students and "
        f"{filtered['Name_of_School'].nunique()} schools.",
    )

    # Bento KPI grid
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1], gap="medium")
    with c1:
        kpi("Pass rate", f"{pass_rate:.1f}%",
            delta=f"{passed:,} of {total:,} qualified",
            accent="var(--accent)")
    with c2:
        kpi("Average score", f"{avg:.1f}",
            sub="out of 200", accent="var(--ink)")
    with c3:
        kpi("Schools", f"{filtered['Name_of_School'].nunique()}",
            sub="analysed", accent="#0891B2")
    with c4:
        kpi("Absent", f"{int(filtered['Absent'].sum())}",
            sub="excluded from remedial", accent="#D97706")

    # ── Distribution ─────────────────────────────────────
    section("Score distribution")

    c_left, c_right = st.columns([2, 1], gap="medium")

    with c_left:
        sample = filtered["Average"].dropna()
        nbins = 28

        fig = px.histogram(
            filtered, x="Average", nbins=nbins,
            color_discrete_sequence=["#6D28D9"], opacity=0.88,
        )

        # ── KDE overlay (optional, may fail on degenerate samples) ──
        if len(sample) > 5 and sample.nunique() > 1:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(sample)
                xs = np.linspace(sample.min(), sample.max(), 200)
                bin_w = (sample.max() - sample.min()) / nbins
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=kde(xs) * len(sample) * bin_w,
                    mode="lines",
                    line=dict(color="#0B0B0F", width=2),
                    name="Density",
                    hoverinfo="skip",
                ))
            except (np.linalg.LinAlgError, ValueError):
                pass  # skip KDE if the sample is singular / too small

        fig.add_vline(
            x=35, line_dash="dash", line_color="#B91C1C",
            annotation_text="Pass · 35", annotation_position="top",
        )
        fig.add_vline(
            x=avg, line_dash="dot", line_color="#047857",
            annotation_text=f"Mean · {avg:.1f}",
            annotation_position="top right",
        )
        fig.update_layout(
            title="Distribution of average scores",
            bargap=0.04, showlegend=False,
            height=420,
        )
        st.plotly_chart(fig, width="stretch")

    with c_right:
        fail = total - passed
        fig = go.Figure(go.Pie(
            values=[passed, fail],
            labels=["Qualified", "Remedial"],
            hole=0.80,
            marker=dict(
                colors=["#047857", "#E5E2DB"],
                line=dict(color="#FBFAF7", width=4),
            ),
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            sort=False,
        ))
        fig.add_annotation(
            text=(f"<b style='font-size:28px;color:#0B0B0F'>{pass_rate:.1f}%</b>"
                  f"<br><span style='font-size:10px;letter-spacing:.16em;"
                  f"color:#64646E'>PASS RATE</span>"),
            x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(
            title="Qualification",
            showlegend=True,
            legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
            height=420,
            margin=dict(l=20, r=20, t=48, b=20),
        )
        st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════
# PAGE 2 — SCHOOLS & ZONES
# ════════════════════════════════════════════════════════════
elif page == "Schools & Zones":
    page_head(
        "Comparative analytics",
        "Schools & Zones",
        "Rank institutions by qualification rate, above-70 performance, "
        "and target progress.",
    )

    perf = data["school_perf"]
    if selected_zone != "All":
        perf = perf[perf["Name_of_School"].isin(
            filtered["Name_of_School"].unique()
        )]

    # ── Editorial leaderboard ─────────────────────────────
    section("Leaderboard")

    top = perf.nlargest(20, "P_70").reset_index(drop=True)
    rows_html = ""
    for i, r in top.iterrows():
        rank_cls = {0: "gold", 1: "silver", 2: "bronze"}.get(i, "")
        width = min(max(float(r["P_70"]), 0), 100)
        rows_html += (
            f'<div class="lb-row">'
            f'<div class="lb-rank {rank_cls}">{i+1:02d}</div>'
            f'<div class="lb-name">{html.escape(str(r["Name_of_School"]))}</div>'
            f'<div class="lb-bar"><div class="lb-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="lb-val">{r["P_70"]:.1f}%</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="lb">{rows_html}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Showing top 20 of {len(perf)} schools · ranked by qualification rate")

    # ── Zone comparison ───────────────────────────────────
    zone_perf = data["zone_perf"]
    if len(zone_perf) > 1:
        section("Zone comparison")
        fig = px.bar(
            zone_perf.sort_values("P_70"),
            x="P_70", y="Zone", orientation="h",
            color="P_70", color_continuous_scale=["#E8E5DF", "#6D28D9"],
            text="P_70",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            title="Qualification rate by zone",
            coloraxis_showscale=False,
            xaxis_title="Qualification rate (%)",
            yaxis_title="",
            height=340,
        )
        st.plotly_chart(fig, width="stretch")

    # ── Radar: top 5 ──────────────────────────────────────
    section("Multi-dimensional performance")
    top5 = perf.nlargest(5, "P_70")
    cats = ["Qualification %", "Above 70", "Applied", "Satisfied"]
    fig = go.Figure()
    for _, r in top5.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[r["P_70"], r["Above_70"], r["APP"], r["SAT"]],
            theta=cats, fill="toself",
            name=str(r["Name_of_School"])[:24],
            line=dict(width=1.5),
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#F0EDE7",
                            tickfont=dict(size=10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Top 5 schools · 4-dimension profile",
        showlegend=True,
        height=520,
    )
    st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════
# PAGE 3 — STATISTICS
# ════════════════════════════════════════════════════════════
elif page == "Statistics":
    page_head(
        "Diagnostics",
        "Statistical Deep Dive",
        "Distribution shape, correlations, and normality diagnostics.",
    )

    # ── Summary + normality side-by-side ──────────────────
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        section("Descriptive statistics")
        st.dataframe(
            data["stats_summary"].set_index("Statistic"),
            width="stretch", height=400,
        )

    with right:
        section("Normality tests")
        if not data["normality"].empty:
            st.dataframe(
                data["normality"],
                width="stretch", hide_index=True,
            )
            st.caption(
                "p > 0.05 → cannot reject normality · "
                "p ≤ 0.05 → evidence of non-normal distribution."
            )
        else:
            st.warning("Not enough data to perform normality tests.")

    # ── Correlation ───────────────────────────────────────
    section("Feature correlations")
    corr_cols = ["Part_I", "Part_II", "Total", "Average", "Z_Score", "T_Score"]
    corr = filtered[corr_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=["#DC2626", "#FBFAF7", "#6D28D9"],
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(
        title="Pearson correlation matrix",
        height=460,
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Q-Q plot ──────────────────────────────────────────
    section("Q-Q plot · normal probability")
    sample = filtered["Average"].dropna()
    if len(sample) > 10:
        res = stats.probplot(sample, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res[0][0], y=res[0][1],
            mode="markers",
            marker=dict(size=5, color="#6D28D9", opacity=0.65,
                        line=dict(width=0)),
            name="Samples",
        ))
        fig.add_trace(go.Scatter(
            x=res[0][0], y=res[1][0] * res[0][0] + res[1][1],
            mode="lines",
            line=dict(color="#0B0B0F", width=2, dash="dash"),
            name="Reference",
        ))
        fig.update_layout(
            title="Sample quantiles vs theoretical normal",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Sample quantiles",
            height=460,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough samples for a Q-Q plot.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — REMEDIAL
# ════════════════════════════════════════════════════════════
elif page == "Remedial":
    page_head(
        "Interventions",
        "Remedial Targets",
        "Students below the pass threshold who require immediate "
        "academic intervention.",
    )

    rem = data["remedial"]
    if selected_zone != "All":
        rem = rem[rem["Zone"] == selected_zone]
    if selected_school != "All":
        rem = rem[rem["Name_of_School"] == selected_school]

    if rem.empty:
        st.markdown(
            '<div class="empty">No remedial students match the current filters.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    deficit = 35 - rem["Average"].mean()

    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    with c1:
        kpi("Students needing help", f"{len(rem):,}",
            sub=f"of {total:,} filtered", accent="var(--danger)")
    with c2:
        kpi("Avg deficit from pass mark", f"{deficit:.1f}",
            sub="points below 35", accent="#D97706")
    with c3:
        worst = rem["Average"].min()
        kpi("Lowest average", f"{worst:.1f}",
            sub="priority case", accent="var(--ink)")

    # ── Deficit distribution ──────────────────────────────
    section("Where remediation is most needed")

    left, right = st.columns([1, 1], gap="medium")

    with left:
        fig = px.histogram(
            rem, x="Diff_from_35", nbins=20,
            color_discrete_sequence=["#DC2626"], opacity=0.85,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#0B0B0F")
        fig.update_layout(
            title="Deficit distribution (points below 35)",
            xaxis_title="Deficit", yaxis_title="Students",
            bargap=0.04, showlegend=False, height=360,
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        by_school = (
            rem.groupby("Name_of_School").size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(12)
        )
        fig = px.bar(
            by_school.sort_values("Count"),
            x="Count", y="Name_of_School", orientation="h",
            color="Count",
            color_continuous_scale=["#FEE2E2", "#B91C1C"],
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            title="Top 12 schools by remedial count",
            coloraxis_showscale=False,
            xaxis_title="Students", yaxis_title="",
            height=420,
        )
        st.plotly_chart(fig, width="stretch")

    # ── Detailed student table with progress bars ─────────
    section("Priority student list")

    view = (
        rem[["Name_of_Students", "Name_of_School", "Zone",
             "Part_I", "Part_II", "Average", "Diff_from_35"]]
        .sort_values("Average")
        .reset_index(drop=True)
    )

    st.dataframe(
        view,
        width="stretch",
        height=460,
        hide_index=True,
        column_config={
            "Average": st.column_config.ProgressColumn(
                "Average", min_value=0, max_value=100, format="%.1f",
            ),
            "Diff_from_35": st.column_config.NumberColumn(
                "Deficit", format="%.1f pts",
            ),
            "Part_I":  st.column_config.NumberColumn("Part I",  format="%d"),
            "Part_II": st.column_config.NumberColumn("Part II", format="%d"),
        },
    )
    st.caption(
        f"{len(rem):,} students flagged · sorted by lowest average first")
