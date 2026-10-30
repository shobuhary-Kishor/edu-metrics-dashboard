import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="EduMetrics Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F4F6F9; }
    h1, h2, h3 { color: #1E293B; font-weight: 700; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #E2E8F0;
        padding: 15px 10px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1); }
    div[data-testid="stMetricValue"] { color: #0F172A; font-size: 28px; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: #64748B; font-size: 14px; font-weight: 500; }
    section[data-testid="stSidebar"] { background-color: #1E293B; color: #F8FAFC; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #F8FAFC; }
    section[data-testid="stSidebar"] .stRadio label { color: #CBD5E1; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; background-color: #E2E8F0; border-radius: 8px; color: #475569; }
    .stTabs [aria-selected="true"] { background-color: #3B82F6; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DYNAMIC DATA PROCESSING (Robust & Error-Proof)
# ==========================================
@st.cache_data
def process_uploaded_data(uploaded_file):
    """Reads raw Excel, cleans data, and calculates all advanced metrics on the fly."""
    df = pd.read_excel(uploaded_file)
    
    # 1. Clean column names: strip all leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # 2. Flexible Column Mapping to handle Excel variations (e.g., "Part-I", "Part - I", "Part_I")
    column_mapping = {
        "No": "Index", 
        "ZONE": "Zone", 
        "Name of School": "Name_of_School",
        "Name of Students": "Name_of_Students", 
        "Part-I": "Part_I",
        "Part - I": "Part_I",
        "Part_I": "Part_I",
        "Part - II": "Part_II",
        "Part_II": "Part_II",
        "Total": "Total",
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # 3. Validate that required columns exist
    required_cols = ["Part_I", "Part_II", "Total", "Name_of_School", "Name_of_Students", "Zone"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns in the uploaded file: {missing_cols}")
        st.info(f"💡 Available columns in your file: {list(df.columns)}")
        st.stop() # Stop execution to prevent further KeyError crashes

    # 4. Clean Data (convert to numeric, fill NaNs with 0)
    for col in ["Part_I", "Part_II", "Total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
    # 5. Core Metrics
    df["Average"] = (df["Part_I"] + df["Part_II"]) / 2
    df["P1_above_35"] = np.where(df["Part_I"] >= 35, "QP1", "NQP1")
    df["P2_above_35"] = np.where(df["Part_II"] >= 35, "QP2", "NQP2")
    df["above_70"] = np.where((df["Part_I"] >= 35) & (df["Part_II"] >= 35), "Q", "NQ")
    df["Diff_from_35"] = df["Average"] - 35
    
    mean_score = df["Average"].mean()
    std_score = df["Average"].std(ddof=0)
    if std_score == 0 or np.isnan(std_score): std_score = 1
    df["Z_Score"] = (df["Average"] - mean_score) / std_score
    df["T_Score"] = 50 + 10 * df["Z_Score"]
    
    # 6. Remedial Targets
    target_students = df[df["Average"] < 35].copy()
    
    # 7. School Metrics
    school_total = df.groupby("Name_of_School")["Average"].count()
    school_pass = df.assign(_pass=(df["Average"] >= 35).astype(int)).groupby("Name_of_School")["_pass"].sum()
    school_percentage = (school_pass / school_total) * 100
    school_metrics = pd.DataFrame({
        "Total_Students": school_total, "Passed": school_pass, "Current_%": school_percentage
    }).reset_index()
    
    target_pct = 80.0
    school_metrics["Target_Passes"] = (school_metrics["Total_Students"] * target_pct / 100).round().astype(int)
    school_metrics["Needed_Passes"] = (school_metrics["Target_Passes"] - school_metrics["Passed"]).clip(lower=0)
    
    # 8. Exam Dept Analysis (School & Zone Performance)
    df["CLEAN_MARKS"] = pd.to_numeric(df["Total"], errors="coerce")
    bins = [-1, 9, 19, 29, 39, 49, 59, 69, 79, 99, 130, 149, np.inf]
    labels = ["1-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-99", "100-130", "131-149", "150+"]
    df["MARK_RANGE"] = pd.cut(df["CLEAN_MARKS"], bins=bins, labels=labels, include_lowest=True).astype(str).replace("nan", "Absent")
    
    cut_off = 35.0
    school_perf = df.groupby("Name_of_School").agg(
        APP=("Index", "count"), SAT=("CLEAN_MARKS", lambda x: x.notna().sum()),
        Above_70=("CLEAN_MARKS", lambda x: (x >= 70).sum()), Above_CutOff=("CLEAN_MARKS", lambda x: (x >= cut_off).sum()),
    ).reset_index()
    school_perf["P_70"] = np.where(school_perf["SAT"] > 0, ((school_perf["Above_70"] / school_perf["SAT"]) * 100).round(2), 0.0)
    
    zone_perf = df.groupby("Zone").agg(
        APP=("Index", "count"), SAT=("CLEAN_MARKS", lambda x: x.notna().sum()),
        Above_70=("CLEAN_MARKS", lambda x: (x >= 70).sum()), Above_CutOff=("CLEAN_MARKS", lambda x: (x >= cut_off).sum()),
    ).reset_index()
    zone_perf["P_70"] = np.where(zone_perf["SAT"] > 0, ((zone_perf["Above_70"] / zone_perf["SAT"]) * 100).round(2), 0.0)
    
    # 9. Statistical Summary & Normality Tests
    raw_scores = df["Average"].dropna()
    stats_summary = pd.DataFrame({
        "Statistic": ["N", "Mean", "Median", "Std Dev", "Min", "Max", "Q1", "Q3", "Skewness", "Kurtosis"],
        "Value": [len(raw_scores), round(raw_scores.mean(), 3), round(raw_scores.median(), 3), 
                  round(raw_scores.std(), 3), raw_scores.min(), raw_scores.max(), 
                  round(raw_scores.quantile(0.25), 3), round(raw_scores.quantile(0.75), 3), 
                  round(raw_scores.skew(), 3), round(raw_scores.kurtosis(), 3)]
    })
    
    normality_tests = pd.DataFrame()
    if len(raw_scores) >= 8:
        sw_stat, sw_p = stats.shapiro(raw_scores)
        ks_stat, ks_p = stats.kstest(raw_scores, stats.norm.cdf, args=(raw_scores.mean(), raw_scores.std()))
        try: dag_stat, dag_p = stats.normaltest(raw_scores)
        except: dag_stat, dag_p = np.nan, np.nan
        try:
            ad_res = stats.anderson(raw_scores, dist='norm', method='interpolate')
            ad_p = float(getattr(ad_res, 'pvalue', np.nan))
        except:
            ad_res = stats.anderson(raw_scores, dist='norm'); ad_p = np.nan
            
        normality_tests = pd.DataFrame({
            "Test": ["Shapiro-Wilk", "Kolmogorov-Smirnov", "D'Agostino-Pearson", "Anderson-Darling"],
            "Statistic": [round(sw_stat, 4), round(ks_stat, 4), round(dag_stat, 4) if not np.isnan(dag_stat) else "N/A", round(ad_res.statistic, 4)],
            "p-value": [round(sw_p, 4), round(ks_p, 4), round(dag_p, 4) if not np.isnan(dag_p) else "N/A", round(ad_p, 4) if not np.isnan(ad_p) else "N/A"],
            "Conclusion": [
                "Normal" if sw_p > 0.05 else "Not Normal", "Normal" if ks_p > 0.05 else "Not Normal",
                "Normal" if (not np.isnan(dag_p) and dag_p > 0.05) else "Not Normal",
                "Normal" if (not np.isnan(ad_p) and ad_p > 0.05) else "Not Normal"
            ]
        })

    return {
        "students": df, "school_metrics": school_metrics, "remedial": target_students,
        "school_perf": school_perf, "zone_perf": zone_perf,
        "stats_summary": stats_summary, "normality": normality_tests
    }

# ==========================================
# SIDEBAR: UPLOAD & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 🎓 EduMetrics")
    st.markdown("### Analytics Dashboard")
    st.divider()
    
    # 🚀 DYNAMIC FILE UPLOADER
    uploaded_file = st.file_uploader("Upload Raw Data (G05_ExamB_Analysis.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        with st.spinner("Processing data & calculating statistics..."):
            data = process_uploaded_data(uploaded_file)
        if data is None:
            st.stop()
        st.success("✅ Data loaded successfully!")
    else:
        st.info("👆 Please upload the raw Excel file to begin.")
        st.stop()
        
    st.divider()
    page = st.radio("Navigation", ["📊 Executive Overview", "🏫 School & Zone Analytics", "📈 Statistical Deep Dive", "🎯 Remedial Targets"])
    
    st.divider()
    st.markdown("### ⚙️ Global Filters")
    
    zones = ["All"] + sorted(data["students"]["Zone"].dropna().unique().tolist())
    selected_zone = st.selectbox("Select Zone", zones)
    
    filtered_students = data["students"]
    if selected_zone != "All":
        filtered_students = filtered_students[filtered_students["Zone"] == selected_zone]
        
    schools = ["All"] + sorted(filtered_students["Name_of_School"].dropna().unique().tolist())
    selected_school = st.selectbox("Select School", schools)
    
    if selected_school != "All":
        filtered_students = filtered_students[filtered_students["Name_of_School"] == selected_school]

    st.divider()
    st.markdown("### 📥 Export Data")
    csv = filtered_students.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered CSV", csv, "filtered_students.csv", "text/csv")

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    total_students = len(filtered_students)
    passed_students = len(filtered_students[filtered_students['above_70'] == 'Q'])
    pass_rate = (passed_students / total_students * 100) if total_students > 0 else 0
    avg_score = filtered_students['Average'].mean() if total_students > 0 else 0
    
    col1.metric("Total Students", f"{total_students:,}")
    col2.metric("Qualified (Pass)", f"{passed_students:,}", delta=f"{pass_rate:.1f}%")
    col3.metric("Average Score", f"{avg_score:.2f}")
    col4.metric("Schools Analyzed", filtered_students['Name_of_School'].nunique())
    
    st.divider()
    tab1, tab2 = st.tabs(["Score Distribution", "Pass vs Fail Composition"])
    
    with tab1:
        fig_hist = px.histogram(filtered_students, x="Average", nbins=30, color_discrete_sequence=['#3B82F6'],
                                title="Distribution of Student Average Scores", template="plotly_white")
        fig_hist.add_vline(x=35, line_dash="dash", line_color="red", annotation_text="Pass Threshold (35)")
        fig_hist.add_vline(x=avg_score, line_dash="dot", line_color="green", annotation_text=f"Mean ({avg_score:.1f})")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with tab2:
        fail_count = total_students - passed_students
        fig_pie = px.pie(values=[passed_students, fail_count], names=['Qualified', 'Needs Remedial'],
                         color_discrete_sequence=['#10B981', '#EF4444'], hole=0.5, title="Overall Qualification Status")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# PAGE 2: SCHOOL & ZONE ANALYTICS
# ==========================================
elif page == "🏫 School & Zone Analytics":
    st.markdown("# 🏫 School & Zone Analytics")
    
    perf_data = data["school_perf"]
    if selected_zone != "All":
        perf_data = perf_data[perf_data["Name_of_School"].isin(filtered_students["Name_of_School"].unique())]
        
    top_schools = perf_data.nlargest(10, 'P_70')
    fig_bar = px.bar(top_schools, x='P_70', y='Name_of_School', orientation='h', color='P_70', 
                     color_continuous_scale='Blues', title="Top 10 Schools by Qualification Rate (%)", template="plotly_white")
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()
    st.markdown("### 🕸️ Top 5 Schools Performance Radar")
    top5 = perf_data.nlargest(5, 'P_70')
    categories = ['Qualification %', 'Above 70 Marks', 'Total Applied', 'Total Satisfied']
    fig_radar = go.Figure()
    
    for idx, row in top5.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['P_70'], row['Above_70'], row['APP'], row['SAT']],
            theta=categories, fill='toself', name=str(row['Name_of_School'])[:25]
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Multi-Dimensional School Performance")
    st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# PAGE 3: STATISTICAL DEEP DIVE
# ==========================================
elif page == "📈 Statistical Deep Dive":
    st.markdown("# 📈 Statistical Deep Dive")
    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Correlation Matrix", "Normality Tests"])
    
    with tab1:
        st.markdown("### Descriptive Statistics")
        st.dataframe(data["stats_summary"].set_index('Statistic'), use_container_width=True, height=400)
        
    with tab2:
        st.markdown("### Correlation Heatmap")
        corr_cols = ['Part_I', 'Part_II', 'Total', 'Average', 'Z_Score', 'T_Score']
        corr_matrix = filtered_students[corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto", title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with tab3:
        st.markdown("### Normality Test Results")
        if not data["normality"].empty:
            st.dataframe(data["normality"], use_container_width=True, hide_index=True)
        else:
            st.warning("Not enough data to perform normality tests.")
        
        st.markdown("### Q-Q Plot (Normal Distribution Check)")
        sample_data = filtered_students['Average'].dropna()
        if len(sample_data) > 10:
            res = stats.probplot(sample_data, dist="norm")
            fig_qq = px.scatter(x=res[0][0], y=res[0][1], labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}, title="Q-Q Plot")
            fig_qq.add_trace(go.Scatter(x=res[0][0], y=res[1][0]*res[0][0] + res[1][1], mode='lines', line=dict(color='red', dash='dash'), name='Trendline'))
            st.plotly_chart(fig_qq, use_container_width=True)

# ==========================================
# PAGE 4: REMEDIAL TARGETS
# ==========================================
elif page == "🎯 Remedial Targets":
    st.markdown("# 🎯 Remedial Targets & Interventions")
    st.markdown("Students scoring below the threshold requiring immediate academic intervention.")
    
    remedial_df = data["remedial"]
    if selected_zone != "All": remedial_df = remedial_df[remedial_df['Zone'] == selected_zone]
    if selected_school != "All": remedial_df = remedial_df[remedial_df['Name_of_School'] == selected_school]
        
    col1, col2 = st.columns(2)
    with col1: st.metric("Total Students Needing Help", len(remedial_df))
    with col2:
        avg_deficit = 35 - remedial_df['Average'].mean() if len(remedial_df) > 0 else 0
        st.metric("Average Deficit from Pass Mark", f"{avg_deficit:.1f} points", delta=f"-{avg_deficit:.1f}", delta_color="inverse")
        
    st.divider()
    school_remedial = remedial_df.groupby('Name_of_School').size().reset_index(name='Count').sort_values('Count', ascending=False).head(15)
    fig_rem = px.bar(school_remedial, x='Name_of_School', y='Count', color='Count', color_continuous_scale='Reds',
                     title="Top 15 Schools with Highest Remedial Requirements", template="plotly_white")
    st.plotly_chart(fig_rem, use_container_width=True)
    
    st.markdown("### 📋 Detailed Remedial Student List")
    st.dataframe(remedial_df[['Name_of_Students', 'Name_of_School', 'Part_I', 'Part_II', 'Average', 'Diff_from_35']], use_container_width=True, height=400)
