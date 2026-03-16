"""
CO₂ Policy Intelligence Dashboard
Professional editorial theme inspired by Our World in Data
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import joblib
import time
import google.generativeai as genai

#  PAGE CONFIG 
st.set_page_config(
    page_title="CO₂ Policy Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  DESIGN SYSTEM
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Global ── */
html, body, .stApp {
    background: #fafaf8 !important;
    font-family: 'Source Sans 3', sans-serif;
    color: #1a1a1a;
}
[data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid #e8e5de !important;
}

/* ── Top nav bar ── */
.top-nav {
    background: #fff;
    border-bottom: 2px solid #c0392b;
    padding: 14px 0 12px;
    margin-bottom: 0;
}
.nav-inner {
    display: flex; align-items: center;
    justify-content: space-between;
    max-width: 1200px; margin: 0 auto;
    padding: 0 24px;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 20px; font-weight: 700;
    color: #1a1a1a; letter-spacing: -.3px;
}
.nav-logo span { color: #c0392b; }
.nav-tagline {
    font-size: 12px; color: #666;
    font-weight: 300; letter-spacing: .02em;
}

/* ── Hero section ── */
.hero {
    background: linear-gradient(135deg, #1a2744 0%, #2c3e6e 60%, #1a3a4a 100%);
    padding: 52px 40px 44px;
    margin-bottom: 32px;
    border-radius: 0;
}
.hero-label {
    font-size: 11px; font-weight: 600;
    color: #7eb8d4; text-transform: uppercase;
    letter-spacing: .12em; margin-bottom: 10px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 38px; font-weight: 700;
    color: #fff; line-height: 1.2;
    margin-bottom: 12px; max-width: 640px;
}
.hero-sub {
    font-size: 16px; color: #b8c8d8;
    line-height: 1.6; max-width: 560px;
    font-weight: 300;
}
.hero-stats {
    display: flex; gap: 36px;
    margin-top: 28px; flex-wrap: wrap;
}
.hero-stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 28px; color: #fff; font-weight: 700;
}
.hero-stat-lbl {
    font-size: 12px; color: #7eb8d4;
    font-weight: 400; margin-top: 2px;
}

/* ── Section headers (editorial style) ── */
.section-head {
    border-left: 4px solid #c0392b;
    padding-left: 16px;
    margin-bottom: 18px;
}
.section-head h3 {
    font-family: 'Playfair Display', serif !important;
    font-size: 20px; font-weight: 700;
    color: #1a1a1a !important; margin: 0 0 3px;
}
.section-head p {
    font-size: 13px; color: #666;
    margin: 0; font-weight: 300; line-height: 1.5;
}

/* ── Explanation callout (learn section) ── */
.learn-box {
    background: #f0f4ff;
    border: 1px solid #c5d0f0;
    border-left: 4px solid #3b5bdb;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px 14px 16px;
    margin-bottom: 16px;
    font-size: 13px; color: #2c3e8a;
    line-height: 1.7;
}
.learn-box strong { color: #1a2a6e; }
.learn-icon { font-size: 16px; margin-right: 6px; }

/* ── Metric cards ── */
.metric-card {
    background: #fff;
    border: 1px solid #e8e5de;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: box-shadow .2s;
}
.metric-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.08); }
.metric-label {
    font-size: 11px; font-weight: 600;
    color: #888; text-transform: uppercase;
    letter-spacing: .07em; margin-bottom: 5px;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 22px; color: #1a1a1a; font-weight: 700;
}
.metric-unit { font-size: 12px; color: #aaa; margin-top: 3px; }

/* ── Level badges ── */
.badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 3px; font-size: 12px;
    font-weight: 600; letter-spacing: .04em;
    text-transform: uppercase;
}
.badge-high   { background: #fde8e8; color: #c0392b; border: 1px solid #f5c6c6; }
.badge-medium { background: #fef3e2; color: #d4860a; border: 1px solid #f5dba0; }
.badge-low    { background: #e8f5ee; color: #1e7c45; border: 1px solid #a8d8bc; }
.badge-na     { background: #f5f5f5; color: #888;    border: 1px solid #ddd; }

/* ── Comparison card (ML vs IPCC) ── */
.compare-row {
    display: flex; gap: 10px;
    align-items: stretch; margin-bottom: 12px;
}
.compare-half {
    flex: 1; background: #fff;
    border: 1px solid #e8e5de;
    border-radius: 8px; padding: 14px 16px;
}
.compare-half-label {
    font-size: 10px; font-weight: 600;
    color: #888; text-transform: uppercase;
    letter-spacing: .07em; margin-bottom: 8px;
}
.compare-half-name {
    font-size: 12px; color: #444;
    margin-bottom: 6px; font-weight: 400;
}

/* ── Feature bar ── */
.feat-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.feat-rank { font-size:11px; color:#aaa; width:18px; text-align:right; flex-shrink:0; }
.feat-name { font-size:12px; color:#333; width:128px; flex-shrink:0; }
.feat-bg { flex:1; height:14px; background:#f0ede8; border-radius:2px; overflow:hidden; }
.feat-val { font-size:11px; color:#888; width:38px; text-align:right; flex-shrink:0; }

/* ── Policy box ── */
.policy-card {
    background: #fff;
    border: 1px solid #e8e5de;
    border-top: 3px solid #1e7c45;
    border-radius: 0 0 8px 8px;
    padding: 16px 18px;
    font-size: 13px; color: #333; line-height: 1.8;
}
.policy-card.medium { border-top-color: #d4860a; }
.policy-card.high   { border-top-color: #c0392b; }

/* ── AI context box ── */
.ai-box {
    background: #fffdf5;
    border: 1px solid #e8dfc0;
    border-left: 4px solid #d4860a;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px 16px 16px;
    font-size: 13px; color: #444; line-height: 1.85;
    margin-top: 10px;
}
.ai-box-header {
    font-size: 11px; font-weight: 600;
    color: #a06000; text-transform: uppercase;
    letter-spacing: .07em; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}

/* ── Data table ── */
.stDataFrame { border: 1px solid #e8e5de !important; border-radius: 8px !important; }

/* ── Confidence bar ── */
.conf-row { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
.conf-label { font-size:12px; color:#555; width:64px; flex-shrink:0; }
.conf-bg { flex:1; height:16px; background:#f0ede8; border-radius:2px; overflow:hidden; }
.conf-val { font-size:11px; color:#888; width:38px; text-align:right; flex-shrink:0; }

/* ── Sidebar base ── */
[data-testid="stSidebar"] { display: flex !important; flex-direction: column !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #444 !important; font-size:13px; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1a1a1a !important; }
.sidebar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 17px; font-weight: 700; color: #1a1a1a;
    padding-bottom: 12px;
    border-bottom: 1px solid #e8e5de;
    margin-bottom: 6px;
}
.sidebar-logo span { color: #c0392b; }
.sidebar-footer {
    margin-top: auto !important;
    padding-top: 12px !important;
    border-top: 1px solid #e8e5de !important;
}

/* ── Nav radio — 4 rules only ──
   Rule 1: hide the radio circle (the <div> Streamlit draws around it)
   Rule 2: style every label as a plain left-aligned row
   Rule 3: hover state
   Rule 4: selected state — light blue bg, red text ── */

/* Rule 1 — hide circle only (div[data-baseweb] first-child is the dot, NOT the text) */
[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] > div:first-child { display:none !important; }

/* Rule 2 — label = full-width left-aligned row */
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    width: calc(100% + 40px) !important;
    padding: 10px 34px 10px 14px !important;
    margin: 0 -20px 2px 0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #333 !important;
    cursor: pointer !important;
    background: transparent !important;
    box-sizing: border-box !important;
    transition: background .12s !important;
}

/* Rule 3 — hover */
[data-testid="stSidebar"] .stRadio label:hover {
    background: #f0ede8 !important;
    color: #1a1a1a !important;
}

/* Rule 4 — selected */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #eef2fd !important;
    color: #c0392b !important;
    font-weight: 600 !important;
}

/* Remove gap above radio group */
[data-testid="stSidebar"] .stRadio { margin-top: 0 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }

/* Nav divider and API key label */
.nav-divider { height: 1px; background: #e8e5de; margin: 10px 0; }
.nav-api-label {
    font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: .06em;
    padding: 0 2px; margin-bottom: 6px; margin-top: 4px;
}

/* ── Global Streamlit overrides ── */
div[data-testid="stMetricValue"] { color: #1a1a1a !important; }
.stSelectbox label, .stTextInput label {
    color: #555 !important; font-size: 12px !important;
    font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: .05em !important;
}
h1,h2,h3 { color: #1a1a1a !important; }
.stButton>button {
    background: #c0392b; color: #fff;
    border: none; border-radius: 4px;
    font-size: 13px; font-weight: 600;
    letter-spacing: .03em; width: 100%;
    padding: 10px;
}
.stButton>button:hover { background: #a93226 !important; }

/* ── Divider / Source tag ── */
hr { border: none; border-top: 1px solid #e8e5de; margin: 24px 0; }
.source-tag { font-size: 11px; color: #aaa; text-align: right; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


#  CONSTANTS
FEATURES = ['energy_per_capita','gdp','population',
            'coal_co2','oil_co2','gas_co2','methane']
FEAT_LABELS = {
    'energy_per_capita': 'Energy per capita',
    'gdp':               'GDP',
    'population':        'Population',
    'coal_co2':          'Coal CO₂',
    'oil_co2':           'Oil CO₂',
    'gas_co2':           'Gas CO₂',
    'methane':           'Methane',
}
FEAT_COLORS = {
    'coal_co2':          '#c0392b',
    'oil_co2':           '#e67e22',
    'gas_co2':           '#f39c12',
    'methane':           '#8e44ad',
    'gdp':               '#2980b9',
    'population':        '#16a085',
    'energy_per_capita': '#27ae60',
}
LEVEL_COLORS = {'High':'#c0392b','Medium':'#d4860a','Low':'#1e7c45','N/A':'#aaa'}
BINS = (2.05, 28.76)

AI_MODEL = "gemini-2.5-flash"
AI_MODEL     = "gemini-2.5-flash"
MAX_REQUESTS = 5      
COOLDOWN     = 30  

#  DATA LOADING
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df  = pd.read_csv(url)
    df_clean = df[df['year'] >= 1990].copy()
    exclude = [
        'World','Africa','Asia','Europe','Oceania','North America','South America',
        'Africa (GCP)','Asia (GCP)','Europe (GCP)','Oceania (GCP)',
        'North America (GCP)','South America (GCP)',
        'European Union (27)','European Union (28)',
        'Asia (excl. China and India)','Europe (excl. EU-27)',
        'Europe (excl. EU-28)','North America (excl. USA)',
        'High-income countries','Low-income countries',
        'Upper-middle-income countries','Lower-middle-income countries',
        'Least developed countries (Jones et al.)',
        'OECD (GCP)','OECD (Jones et al.)','Non-OECD (GCP)',
        'Central America (GCP)','Middle East (GCP)',
        'International aviation','International shipping',
        'Kuwaiti Oil Fires','Kuwaiti Oil Fires (GCP)',
        'Ryukyu Islands','Ryukyu Islands (GCP)'
    ]
    df_clean = df_clean[~df_clean['country'].isin(exclude)]
    cols = FEATURES + ['co2']
    df_filled = df_clean.copy()
    for col in cols:
        df_filled[col] = df_filled.groupby('country')[col].transform(
            lambda x: x.fillna(x.median()))
    for col in cols:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    df_filled['co2_level'] = pd.qcut(
        df_filled['co2'], q=3, labels=['Low','Medium','High'])
    latest  = df_filled.sort_values('year').groupby('country').last().reset_index()
    history = df_filled[['country','year','co2']].copy()
    return df_filled, latest, history


@st.cache_resource
def load_models():
    dt = joblib.load('decision_tree.pkl')
    rf = joblib.load('random_forest.pkl')
    return dt, rf


#  HELPERS
def ipcc_label(co2_total, population):
    if population and population > 0 and not pd.isna(co2_total) and co2_total > 0:
        pc = (co2_total * 1e6) / population
        if   pc < 3:  return 'Low',    f'{pc:.1f}t/person — within safe range (<3t)'
        elif pc < 10: return 'Medium', f'{pc:.1f}t/person — needs reduction (3–10t)'
        else:          return 'High',   f'{pc:.1f}t/person — urgent action needed (>10t)'
    return 'N/A', 'No per-capita data'


def fmt(val, decimals=2):
    if pd.isna(val) or val is None: return 'N/A'
    if abs(val) >= 1e12: return f'{val/1e12:.{decimals}f}T'
    if abs(val) >= 1e9:  return f'{val/1e9:.{decimals}f}B'
    if abs(val) >= 1e6:  return f'{val/1e6:.{decimals}f}M'
    return f'{val:,.0f}'


def badge(level):
    cls  = {'High':'badge-high','Medium':'badge-medium',
            'Low':'badge-low'}.get(level,'badge-na')
    return f"<span class='badge {cls}'>{level}</span>"


def predict(row, dt_m, rf_m):
    d = pd.DataFrame([{f: row[f] for f in FEATURES}])
    return (dt_m.predict(d)[0],
            rf_m.predict(d)[0],
            dict(zip(rf_m.classes_, rf_m.predict_proba(d)[0])))


def light_chart():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#fafaf8')
    ax.set_facecolor('#fafaf8')
    ax.tick_params(colors='#555', labelsize=10)
    for sp in ax.spines.values(): sp.set_color('#e8e5de')
    ax.grid(axis='both', color='#eeebe6', lw=0.6, linestyle='--')
    return fig, ax

#LIMITER FOR AI REQUESTS
def init_rate_limiter():
    """Set up counters in session state on first load."""
    if 'ai_count' not in st.session_state:
        st.session_state.ai_count = 0
    if 'ai_last_time' not in st.session_state:
        st.session_state.ai_last_time = 0

def check_rate_limit():
    """Returns (allowed=True/False, message)."""
    init_rate_limiter()
    now = time.time()
    if st.session_state.ai_count >= MAX_REQUESTS:
        return False, f"You have used all {MAX_REQUESTS} AI requests this session. Refresh the page to reset."
    elapsed = now - st.session_state.ai_last_time
    if elapsed < COOLDOWN and st.session_state.ai_count > 0:
        wait = int(COOLDOWN - elapsed)
        return False, f"Please wait {wait} more seconds before generating again."
    return True, ""

def record_request():
    """Call after every successful AI call."""
    st.session_state.ai_count += 1
    st.session_state.ai_last_time = time.time()
    
#  AI CONTEXT 
def get_ai_context(country, level, ipcc_lvl, co2_val,
                   gdp_val, population, dominant_fuel):
    # Configure Gemini with your key from secrets.toml
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    # Load the model
    model = genai.GenerativeModel(AI_MODEL)

    prompt = f"""You are a senior climate economist writing for a data journalism platform.

Analyse {country}'s CO₂ emission profile and write a concise structured report.

Data:
- Total CO₂         : {co2_val:.1f} million tonnes/year
- ML classification : {level} emitter
- IPCC standard     : {ipcc_lvl}
- GDP               : {fmt(gdp_val)} USD
- Population        : {fmt(population)}
- Primary source    : {dominant_fuel}

Write exactly 4 short sections:
**Economic Context** (2-3 sentences): Why does this country emit at this level?
**Policy Landscape** (2-3 sentences): What transitions are realistic here?
**Global Comparison** (2 sentences): How does it compare to regional peers?
**Key Risk or Opportunity** (2 sentences): Most important factor next 10 years.

Be specific to {country}. No generic statements."""

    response = model.generate_content(prompt)
    return response.text



#  LOAD
with st.spinner("Loading data..."):
    df_filled, latest, history = load_data()

try:
    dt_model, rf_model = load_models()
    models_ok =   True
    feat_imp  = sorted(zip(FEATURES, rf_model.feature_importances_),
                       key=lambda x: -x[1])
except Exception as e:
    models_ok = False
    feat_imp  = []


#  SIDEBAR
with st.sidebar:

    st.markdown("""
    <div class='sidebar-logo'>CO₂ Policy<span> Intelligence</span></div>
    <p style='font-size:12px;color:#888;margin:6px 0 16px;padding:0'>
      Global emissions · ML · AI advisory
    </p>""", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["Dashboard", "Country Rankings", "Global Charts", "About"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class='sidebar-footer'>
      <p style='margin:0 0 8px 0;font-size:11px;color:#888'><strong>Purpose:</strong> Academic research project</p>
      <p style='margin:0;font-size:11px;color:#888'>Data sourced from <a href='https://ourworldindata.org/' target='_blank' style='color:#2980b9;text-decoration:none'>Our World in Data</a></p>
    </div>""", unsafe_allow_html=True)

#  HERO BANNER  (shown on all pages)
total_high   = len(latest[latest['co2_level']=='High'])
total_medium = len(latest[latest['co2_level']=='Medium'])
total_low    = len(latest[latest['co2_level']=='Low'])

st.markdown(f"""
<div class='hero'>
  <div class='hero-label'>CO₂ Emissions Intelligence Platform</div>
  <div class='hero-title'>Understanding global CO₂ emissions through data and machine learning</div>
  <div class='hero-sub'>Track, classify and analyse the emission profiles of 219 countries using Random Forest and Decision Tree models trained on Our World in Data.</div>
  <div class='hero-stats'>
    <div>
      <div class='hero-stat-val'>219</div>
      <div class='hero-stat-lbl'>Countries analysed</div>
    </div>
    <div>
      <div class='hero-stat-val'>{total_high}</div>
      <div class='hero-stat-lbl'>High emitters</div>
    </div>
    <div>
      <div class='hero-stat-val'>{total_medium}</div>
      <div class='hero-stat-lbl'>Medium emitters</div>
    </div>
    <div>
      <div class='hero-stat-val'>{total_low}</div>
      <div class='hero-stat-lbl'>Low emitters</div>
    </div>
    <div>
      <div class='hero-stat-val'>95.74%</div>
      <div class='hero-stat-lbl'>RF model accuracy</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


#  PAGE 1 — COUNTRY PROFILE/DASHBOARD
if "Dashboard" in page:

    # ── SECTION HEADER + learn box ──
    st.markdown("""
    <div class='section-head'>
      <h3>Country Profile & Classification</h3>
      <p>Select a country to see its full emission profile, ML model classification, IPCC global standard, and AI-generated economic context.</p>
    </div>""", unsafe_allow_html=True)

    countries = sorted(latest['country'].unique().tolist())
    col_sel, col_btn = st.columns([3,1])
    with col_sel:
        sel = st.selectbox("Select country", countries,
                           index=countries.index("Indonesia")
                           if "Indonesia" in countries else 0)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_ai = st.button("✦ Generate Context")

    row  = latest[latest['country'] == sel].iloc[0]
    co2  = row.get('co2', 0) or 0
    pop  = row.get('population', 1) or 1
    gdp  = row.get('gdp', 0) or 0
    yr   = int(row.get('year', 2024))

    if models_ok:
        dt_pred, rf_pred, rf_prob = predict(row, dt_model, rf_model)
    else:
        dt_pred = rf_pred = str(row['co2_level'])
        rf_prob = {'Low':0.33,'Medium':0.33,'High':0.34}

    ipcc_lvl, ipcc_desc = ipcc_label(co2, pop)

    dom_fuel = max(['coal_co2','oil_co2','gas_co2'],
                   key=lambda x: row.get(x,0) or 0)
    dom_label = {'coal_co2':'Coal','oil_co2':'Oil','gas_co2':'Natural Gas'}[dom_fuel]

    st.divider()

    # ── Country headline ──
    hc1, hc2, hc3 = st.columns([2,1,1])
    with hc1:
        st.markdown(f"<h2 style='font-family:Playfair Display,serif;font-size:32px;margin-bottom:4px'>{sel}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#888;font-size:13px'>Latest data: {yr} &nbsp;·&nbsp; Total CO₂: <strong style='color:#1a1a1a'>{co2:.2f} Mt/year</strong> &nbsp;·&nbsp; Primary source: <strong style='color:#1a1a1a'>{dom_label}</strong></p>", unsafe_allow_html=True)
    with hc2:
        st.markdown(f"<div style='text-align:center;padding-top:8px'><div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>ML Classification</div>{badge(rf_pred)}</div>", unsafe_allow_html=True)
    with hc3:
        st.markdown(f"<div style='text-align:center;padding-top:8px'><div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>IPCC Standard</div>{badge(ipcc_lvl)}</div>", unsafe_allow_html=True)

    st.divider()

    # ── 3 column layout ──
    col1, col2, col3 = st.columns([1, 1, 1.1])

    # ── COL 1: Key metrics ──
    with col1:
        st.markdown("""
        <div class='section-head'>
          <h3>Key Metrics</h3>
          <p>Core economic and emissions indicators</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='learn-box' style='font-size:12px'>
          <strong>The Key : </strong> These metrics together explain
          the <em>scale</em> (population, GDP) and <em>intensity</em>
          (CO₂ total, energy per capita) of emissions.
          A country with high GDP but low CO₂ is decoupling growth from emissions — a key policy goal.
        </div>""", unsafe_allow_html=True)

        for label, key, unit in [
            ("Total CO₂",         'co2',               "million tonnes / year"),
            ("GDP",               'gdp',               "USD (current prices)"),
            ("Population",        'population',        "people"),
            ("Energy per capita", 'energy_per_capita', "kWh per person / year"),
        ]:
            val = row.get(key)
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value'>{fmt(val)}</div>
              <div class='metric-unit'>{unit}</div>
            </div>""", unsafe_allow_html=True)

    # ── COL 2: Classification + emissions breakdown ──
    with col2:
        st.markdown("""
        <div class='section-head'>
          <h3>Classification</h3>
          <p>ML model vs IPCC global standard</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='learn-box' style='font-size:12px'>
          <strong>ML vs IPCC :</strong> The ML label ranks countries
          relative to each other (bottom/middle/top 33%). The IPCC label
          uses a fixed per-capita threshold rooted in climate science.
          They can disagree — a large country may be ML-High but IPCC-Medium if its population is huge.
        </div>""", unsafe_allow_html=True)

        # classification comparison
        st.markdown(f"""
        <div class='compare-row'>
          <div class='compare-half'>
            <div class='compare-half-label'>Random Forest</div>
            <div class='compare-half-name'>Relative ranking</div>
            {badge(rf_pred)}
          </div>
          <div class='compare-half'>
            <div class='compare-half-label'>Decision Tree</div>
            <div class='compare-half-name'>Interpretable rules</div>
            {badge(dt_pred)}
          </div>
        </div>
        <div class='metric-card' style='margin-bottom:14px'>
          <div class='metric-label'>IPCC Global Standard (per capita)</div>
          <div style='margin-top:4px'>{badge(ipcc_lvl)}</div>
          <div class='metric-unit' style='margin-top:6px'>{ipcc_desc}</div>
        </div>""", unsafe_allow_html=True)

        # RF Confidence
        st.markdown("""
        <div class='section-head' style='margin-top:4px'>
          <h3>Model Confidence</h3>
          <p>Probability assigned by Random Forest</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='learn-box' style='font-size:12px'>
          <strong>Confidence bars:</strong> How certain the RF model is.
          Values near 100% mean the country's features clearly belong to one class.
          Values near 50/50 mean the country sits near a boundary — genuinely ambiguous.
        </div>""", unsafe_allow_html=True)

        for cls in ['Low','Medium','High']:
            prob = rf_prob.get(cls, 0)
            clr  = LEVEL_COLORS.get(cls, '#888')
            op   = 1.0 if cls == rf_pred else 0.25
            st.markdown(f"""
            <div class='conf-row'>
              <span class='conf-label'>{cls}</span>
              <div class='conf-bg'>
                <div style='width:{prob*100:.1f}%;height:100%;background:{clr};opacity:{op};border-radius:2px'></div>
              </div>
              <span class='conf-val'>{prob*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        # Emission sources
        st.markdown("""
        <div class='section-head' style='margin-top:14px'>
          <h3>Emission Sources</h3>
          <p>CO₂ by fuel type (million tonnes)</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='learn-box' style='font-size:12px'>
          <strong>Source For Emission</strong> shows which variables the Random Forest
          relied on most to make its classification. Oil CO₂ ranks #1 (39.8%) —
          meaning oil usage is the strongest single predictor of a country's emission category globally.
          This is <em>global importance</em>, not specific to this country.
        </div>""", unsafe_allow_html=True)

        sources = {
            'Coal':   row.get('coal_co2',0) or 0,
            'Oil':    row.get('oil_co2',0)  or 0,
            'Gas':    row.get('gas_co2',0)  or 0,
            'Methane':row.get('methane',0)  or 0,
        }
        src_clr = {'Coal':'#c0392b','Oil':'#e67e22','Gas':'#f39c12','Methane':'#8e44ad'}
        mx = max(sources.values()) if max(sources.values()) > 0 else 1
        for src, val in sources.items():
            pct = val / mx * 100
            st.markdown(f"""
            <div class='feat-row'>
              <span class='feat-name'>{src}</span>
              <div class='feat-bg'>
                <div style='width:{pct:.1f}%;height:100%;background:{src_clr[src]};border-radius:2px'></div>
              </div>
              <span class='feat-val'>{val:.1f}</span>
            </div>""", unsafe_allow_html=True)

    # ── COL 3: Feature importance + policy ──
    with col3:
        st.markdown("""
        <div class='section-head'>
          <h3>Feature Importance Ranking</h3>
          <p>Which inputs drive the RF model most</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='learn-box' style='font-size:12px'>
          <strong>Feature importance</strong> shows which variables the Random Forest
          relied on most to make its classification. Oil CO₂ ranks #1 (39.8%) —
          meaning oil usage is the strongest single predictor of a country's emission category globally.
          This is <em>global importance</em>, not specific to this country.
        </div>""", unsafe_allow_html=True)

        if models_ok:
            max_imp = feat_imp[0][1]
            for rank, (feat, imp_v) in enumerate(feat_imp, 1):
                pct = imp_v / max_imp * 100
                clr = FEAT_COLORS.get(feat,'#aaa')
                st.markdown(f"""
                <div class='feat-row'>
                  <span class='feat-rank'>#{rank}</span>
                  <span class='feat-name'>{FEAT_LABELS[feat]}</span>
                  <div class='feat-bg'>
                    <div style='width:{pct:.1f}%;height:100%;background:{clr};border-radius:2px'></div>
                  </div>
                  <span class='feat-val'>{imp_v:.3f}</span>
                </div>""", unsafe_allow_html=True)

        # Policy recommendation
        st.markdown("""
        <div class='section-head' style='margin-top:16px'>
          <h3>Policy Recommendation</h3>
          <p>Targeted actions based on emission profile</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='learn-box' style='font-size:12px'>
          <strong>Policy Recomendation:</strong> The policy recommendation
          is rule-based — it reads the ML level and identifies the dominant
          fuel source, then returns targeted actions specific to that combination.
          The AI context section below gives broader economic reasoning.
        </div>""", unsafe_allow_html=True)

        dom_fuel_val = row.get(dom_fuel, 0) or 0
        if rf_pred == 'Low':
            policy_html = (
                f"<strong>Situation:</strong> Below 2.05 Mt/year — primary source is {dom_label}.<br><br>"
                "Maintain current energy standards and efficiency programs,<br>"
                "Invest early in renewables to prevent future lock-in,<br>"
                "Apply for climate finance as a low-emitter nation,<br>"
                "Build national monitoring infrastructure proactively<br><br>"
                "<strong>Quick win:</strong> National LED + building energy codes<br>"
                "<strong>Long-term:</strong> EV transition before fossil dependency deepens"
            )
        elif rf_pred == 'Medium':
            policy_html = (
                f"<strong>Situation:</strong> 2.05–28.76 Mt/year — primary source is {dom_label}. "
                "Critical zone: highest risk of crossing into High.<br><br>"
                f"Implement carbon pricing targeting {dom_label} industry,<br>"
                f"Phase out {dom_label} subsidies — redirect to renewables,<br>"
                "Set national net-zero target with 5-year milestones,<br>"
                "Mandate 30% renewable capacity increase within 5 years<br><br>"
                "<strong>Quick win:</strong> Fuel efficiency standards for all vehicles<br>"
                "<strong>Long-term:</strong> 50%+ renewable electricity grid by 2035"
            )
        else:
            policy_html = (
                f"<strong>Situation:</strong> Above 28.76 Mt/year — primary source is {dom_label}. "
                "Urgent structural action required.<br><br>"
                f"Immediate carbon tax on {dom_label} sector,<br>"
                f"Hard annual emissions cap with legal enforcement,<br>"
                f"Close or retrofit oldest {dom_label} plants within 3 years,<br>"
                "Massive public investment in grid-scale renewables and storage<br><br>"
                f"<strong>Quick win:</strong> Ban all new {dom_label} infrastructure permits<br>"
                f"<strong>Long-term:</strong> Full {dom_label} decarbonisation by 2040"
            )

        lvl_cls = rf_pred.lower()
        st.markdown(f"<div class='policy-card {lvl_cls}'>{policy_html}</div>",
                    unsafe_allow_html=True)

    # ═══════════════════════════════
    #  AI ECONOMIC & POLICY CONTEXT
    # ═══════════════════════════════
    st.divider()
    st.markdown("""
    <div class='section-head'>
      <h3>AI Economic & Policy Context</h3>
      <p>Broader economic reasoning generated by AI</p>
    </div>""", unsafe_allow_html=True)
    
    
    st.markdown("""
    <div class='learn-box'>
      <strong>What the AI adds:</strong> The ML model classifies countries based on numerical patterns
      in the training data. It does not know <em>why</em> a country emits at that level,
      what its economic constraints are, or how it compares to regional peers.
      The AI context fills that gap — it uses Claude to generate economic reasoning,
      policy landscape analysis, global comparisons, and key risk/opportunity identification
      specific to each country's real-world situation.
    </div>""", unsafe_allow_html=True)
    

    if run_ai:
      allowed, msg = check_rate_limit()
      if not allowed:
          st.markdown(f"""
          <div style='background:#fef3e2;border:1px solid #f5dba0;
                      border-left:4px solid #d4860a;
                      border-radius:0 8px 8px 0;
                      padding:12px 16px;font-size:13px;color:#8a5500'>
             <strong>Rate limit:</strong> {msg}
          </div>""", unsafe_allow_html=True)
      else:
          with st.spinner(f"Generating AI context for {sel}..."):
              try:
                  ai_text = get_ai_context(
                      sel, rf_pred, ipcc_lvl, co2, gdp, pop, dom_label
                  )                                    
                  st.session_state[f'ai_{sel}'] = ai_text
                  record_request()                     
              except Exception as e:
                  st.error(f"AI error: {e}")

    if f'ai_{sel}' in st.session_state:
        ai_content = st.session_state[f'ai_{sel}']
        # render markdown cleanly
        st.markdown(f"""
        <div class='ai-box'>
          <div class='ai-box-header'>AI Analysis — {sel}</div>
          {ai_content.replace(chr(10), '<br>').replace('**','<strong>').replace('**','</strong>')}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='ai-box' style='color:#aaa;font-style:italic'>
          <div class='ai-box-header' style='color:#c8a060'>✦ AI Analysis — {sel}</div>
          Click <strong style='color:#d4860a'>Generate AI Context</strong> above to get
          Claude's economic and policy analysis for {sel} — including economic context,
          policy landscape, global comparisons, and key risks/opportunities.
        </div>""", unsafe_allow_html=True)

    # Show AI usage tracker
    init_rate_limiter()
    used     = st.session_state.ai_count
    bar_pct  = (used / MAX_REQUESTS * 100) if MAX_REQUESTS > 0 else 0
    bar_clr  = "#c0392b" if used >= MAX_REQUESTS else "#1e7c45"

    st.markdown(f"""
    <div style='font-size:12px;color:#555;background:#f9f7f2;
                border:1px solid #e8e5de;border-radius:6px;
                padding:12px 16px;margin-top:8px'>
      <div style='font-weight:600;color:#1a1a1a;margin-bottom:8px'>
        ✦ AI Analysis Usage
      </div>
      <div style='background:#e8e5de;border-radius:3px;height:5px;margin-bottom:8px'>
        <div style='width:{bar_pct:.0f}%;height:100%;
                    background:{bar_clr};border-radius:3px'></div>
      </div>
      <div style='font-size:12px;margin-bottom:4px'>{used} / {MAX_REQUESTS} requests used this session</div>
      <div style='color:#aaa;font-size:11px'>
        Note: This is a demonstration of AI integration. The analysis is generated by a language model and should be critically evaluated. It may not reflect the full complexity of {sel}'s economic and policy context.
      </div>
    </div>""", unsafe_allow_html=True)

    
    # ── Historical trend ──
    st.divider()
    st.markdown("""
    <div class='section-head'>
      <h3>Historical CO₂ Trend (1990–2024)</h3>
      <p>How this country's total emissions have changed over time</p>
    </div>""", unsafe_allow_html=True)


    hist = history[history['country']==sel][['year','co2']].dropna()
    if len(hist) > 1:
        fig, ax = light_chart()
        fig.set_size_inches(14, 3.5)
        clr = LEVEL_COLORS.get(rf_pred,'#2980b9')
        ax.fill_between(hist['year'], hist['co2'], alpha=0.12, color=clr)
        ax.plot(hist['year'], hist['co2'], color=clr, linewidth=2.5, zorder=5)
        # threshold lines
        ax.axhline(BINS[0], color='#d4860a', lw=1, ls='--', alpha=0.7)
        ax.axhline(BINS[1], color='#c0392b', lw=1, ls='--', alpha=0.7)
        ax.text(hist['year'].min()+0.5, BINS[0]+0.3,
                f'Medium threshold ({BINS[0]}Mt)', color='#d4860a', fontsize=9)
        ax.text(hist['year'].min()+0.5, BINS[1]+0.3,
                f'High threshold ({BINS[1]}Mt)', color='#c0392b', fontsize=9)
        ax.set_xlabel('Year', color='#555', fontsize=11)
        ax.set_ylabel('CO₂ (million tonnes)', color='#555', fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown(f"<div class='source-tag'>Source: Our World in Data · CO₂ and Greenhouse Gas Emissions dataset</div>", unsafe_allow_html=True)


#  PAGE 2 — COUNTRY RANKINGS
elif "Country Rankings" in page:

    st.markdown("""
    <div class='section-head'>
      <h3>Country Rankings Table</h3>
      <p>All 219 countries with ML classification, IPCC standard, and emission data. Filter by level or search by name.</p>
    </div>""", unsafe_allow_html=True)


    # Filters
    fc1, fc2, fc3 = st.columns([1,1,2])
    with fc1:
        ml_f  = st.multiselect("ML Level",   ['Low','Medium','High'], default=['Low','Medium','High'])
    with fc2:
        ip_f  = st.multiselect("IPCC Level", ['Low','Medium','High','N/A'], default=['Low','Medium','High','N/A'])
    with fc3:
        srch  = st.text_input("Search country name", placeholder="e.g. Indonesia, Brazil, Kenya...")

    # Build table
    rows = []
    for _, r in latest.iterrows():
        c2 = r.get('co2',0) or 0
        pp = r.get('population',1) or 1
        if models_ok:
            d  = pd.DataFrame([{f: r[f] for f in FEATURES}])
            mp = rf_model.predict(d)[0]
            cf = max(rf_model.predict_proba(d)[0])
        else:
            mp = str(r['co2_level']); cf = 0.0
        il, id2 = ipcc_label(c2, pp)
        rows.append({
            'Country':     r['country'],
            'CO₂ (Mt)':   round(c2,2),
            'ML Level':   mp,
            'RF Conf.':   f"{cf*100:.1f}%",
            'IPCC Level': il,
            'IPCC Detail':id2,
            'Coal':       round(r.get('coal_co2',0) or 0,1),
            'Oil':        round(r.get('oil_co2',0)  or 0,1),
            'Gas':        round(r.get('gas_co2',0)  or 0,1),
            'GDP':        fmt(r.get('gdp')),
            'Population': fmt(r.get('population')),
        })

    tdf = pd.DataFrame(rows)
    tdf = tdf[tdf['ML Level'].isin(ml_f)]
    tdf = tdf[tdf['IPCC Level'].isin(ip_f)]
    if srch:
        tdf = tdf[tdf['Country'].str.contains(srch, case=False, na=False)]
    tdf = tdf.sort_values('CO₂ (Mt)', ascending=False).reset_index(drop=True)

    # Summary
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("Countries shown", len(tdf))
    with m2: st.metric("🔴 High",  len(tdf[tdf['ML Level']=='High']))
    with m3: st.metric("🟡 Medium",len(tdf[tdf['ML Level']=='Medium']))
    with m4: st.metric("🟢 Low",   len(tdf[tdf['ML Level']=='Low']))

    def clr_level(val):
        return {
            'High':   'color:#c0392b;font-weight:600',
            'Medium': 'color:#d4860a;font-weight:600',
            'Low':    'color:#1e7c45;font-weight:600',
        }.get(val,'color:#888')

    show = ['Country','CO₂ (Mt)','ML Level','RF Conf.','IPCC Level','IPCC Detail','Coal','Oil','Gas','GDP','Population']
    styled = (tdf[show].style
              .applymap(clr_level, subset=['ML Level','IPCC Level'])
              .format({'CO₂ (Mt)':'{:.2f}','Coal':'{:.1f}','Oil':'{:.1f}','Gas':'{:.1f}'}))
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)
    st.markdown("<div class='source-tag'>Sorted by total CO₂ descending · Source: Our World in Data</div>", unsafe_allow_html=True)

    # ML vs IPCC disagreements
    st.divider()
    st.markdown("""
    <div class='section-head'>
      <h3>ML vs IPCC Disagreements</h3>
      <p>Countries where total CO₂ ranking and per-capita standard give different classifications</p>
    </div>""", unsafe_allow_html=True)


    dis = tdf[tdf['ML Level'] != tdf['IPCC Level']][
        ['Country','CO₂ (Mt)','ML Level','IPCC Level','IPCC Detail']
    ].head(20)
    if len(dis):
        st.dataframe(dis.style.applymap(clr_level, subset=['ML Level','IPCC Level']),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No disagreements in current filter selection.")


#  PAGE 3 — GLOBAL CHARTS
elif "Global Charts" in page:

    st.markdown("""
    <div class='section-head'>
      <h3>Global Emissions Charts</h3>
      <p>Visualising CO₂ patterns across all 219 countries — by total emissions, per capita, fuel source, and ML classification.</p>
    </div>""", unsafe_allow_html=True)


    # build all predictions
    all_out = []
    for _, r in latest.iterrows():
        c2 = r.get('co2',0) or 0
        pp = r.get('population',1) or 1
        mp = rf_model.predict(pd.DataFrame([{f: r[f] for f in FEATURES}]))[0] if models_ok else str(r['co2_level'])
        il, _ = ipcc_label(c2, pp)
        all_out.append({
            'country':   r['country'],
            'co2':       c2,
            'ml_level':  mp,
            'ipcc_level':il,
            'coal_co2':  r.get('coal_co2',0) or 0,
            'oil_co2':   r.get('oil_co2',0)  or 0,
            'gas_co2':   r.get('gas_co2',0)  or 0,
            'co2_pc':    (c2*1e6)/pp if pp > 0 else 0,
        })
    adf = pd.DataFrame(all_out)

    BG   = '#fafaf8'
    GRID = '#eeebe6'
    TICK = '#555'

    def mk_ax(figsize=(10,5)):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.tick_params(colors=TICK, labelsize=10)
        for sp in ax.spines.values(): sp.set_color('#ddd')
        ax.grid(axis='both', color=GRID, lw=0.6, ls='--')
        return fig, ax

    # ── Chart 1 & 2: Pie charts ──
    st.markdown("""
    <div class='section-head'>
      <h3>Classification Distribution</h3>
      <p>How ML model labels (relative) and IPCC labels (per-capita) distribute across 219 countries</p>
    </div>""", unsafe_allow_html=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        cnt = adf['ml_level'].value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        clrs = [LEVEL_COLORS.get(l,'#888') for l in cnt.index]
        wedges, texts, ats = ax.pie(
            cnt.values, labels=cnt.index, colors=clrs,
            autopct='%1.1f%%', pctdistance=0.75, startangle=90,
            wedgeprops={'edgecolor':BG,'linewidth':2.5}
        )
        for t in texts: t.set_fontsize(13); t.set_color('#1a1a1a')
        for t in ats:   t.set_fontsize(11); t.set_color('#fff'); t.set_fontweight('600')
        ax.set_title('ML Model — Total CO₂ ranking', fontsize=12, color='#1a1a1a', pad=12, fontfamily='serif')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with pc2:
        ic = adf['ipcc_level'].value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        iclrs = [LEVEL_COLORS.get(l,'#888') for l in ic.index]
        wedges, texts, ats = ax.pie(
            ic.values, labels=ic.index, colors=iclrs,
            autopct='%1.1f%%', pctdistance=0.75, startangle=90,
            wedgeprops={'edgecolor':BG,'linewidth':2.5}
        )
        for t in texts: t.set_fontsize(13); t.set_color('#1a1a1a')
        for t in ats:   t.set_fontsize(11); t.set_color('#fff'); t.set_fontweight('600')
        ax.set_title('IPCC Standard — Per capita emissions', fontsize=12, color='#1a1a1a', pad=12, fontfamily='serif')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='source-tag'>ML uses pd.qcut equal thirds · IPCC uses 3t/person and 10t/person thresholds</div>", unsafe_allow_html=True)
    st.divider()

    # ── Chart 3: Top 20 total ──
    st.markdown("""
    <div class='section-head'>
      <h3>Top 20 Countries — Total CO₂ Emissions</h3>
      <p>Bars colored by ML classification. Dashed line shows the High threshold (28.76 Mt).</p>
    </div>""", unsafe_allow_html=True)

    top20 = adf.nlargest(20,'co2').sort_values('co2')
    fig, ax = mk_ax((12,7))
    bclrs = [LEVEL_COLORS.get(l,'#888') for l in top20['ml_level']]
    bars = ax.barh(top20['country'], top20['co2'], color=bclrs,
                   edgecolor='white', linewidth=0.5, height=0.65)
    ax.axvline(BINS[1], color='#c0392b', lw=1.2, ls='--', alpha=0.7)
    ax.text(BINS[1]+80, 1, f'High threshold\n({BINS[1]:.0f}Mt)',
            color='#c0392b', fontsize=8.5)
    ax.set_xlabel('CO₂ emissions (million tonnes / year)', color=TICK, fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
    patches = [mpatches.Patch(color=LEVEL_COLORS[l],label=l) for l in ['High','Medium','Low']]
    ax.legend(handles=patches, fontsize=10, framealpha=0.9,
              facecolor=BG, edgecolor='#ddd')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("<div class='source-tag'>Most recent data per country (2024) · Source: Our World in Data</div>", unsafe_allow_html=True)
    st.divider()

    # ── Chart 4: Stacked source breakdown ──
    st.markdown("""
    <div class='section-head'>
      <h3>Emission Sources — Top 15 Countries</h3>
      <p>Stacked breakdown of coal, oil, and gas contributions to total CO₂</p>
    </div>""", unsafe_allow_html=True)

    top15 = adf.nlargest(15,'co2').sort_values('co2')
    fig, ax = mk_ax((12,6))
    y = np.arange(len(top15))
    ax.barh(y, top15['coal_co2'], color='#c0392b', label='Coal', edgecolor='white', lw=0.5, height=0.6)
    ax.barh(y, top15['oil_co2'],  left=top15['coal_co2'], color='#e67e22', label='Oil', edgecolor='white', lw=0.5, height=0.6)
    ax.barh(y, top15['gas_co2'],  left=top15['coal_co2']+top15['oil_co2'], color='#f1c40f', label='Gas', edgecolor='white', lw=0.5, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(top15['country'], fontsize=10)
    ax.set_xlabel('CO₂ (million tonnes)', color=TICK, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9, facecolor=BG, edgecolor='#ddd')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("<div class='source-tag'>Remaining CO₂ (not shown) = other fossil sources · Source: Our World in Data</div>", unsafe_allow_html=True)
    st.divider()

    # ── Chart 5: Historical top 5 trend ──
    st.markdown("""
    <div class='section-head'>
      <h3>Historical Trend — Top 5 Emitters (1990–2024)</h3>
      <p>How the world's largest CO₂ emitters have changed since 1990</p>
    </div>""", unsafe_allow_html=True)

    top5 = adf.nlargest(5,'co2')['country'].tolist()
    fig, ax = mk_ax((12,4.5))
    pal = ['#c0392b','#e67e22','#2980b9','#8e44ad','#16a085']
    for i, ctry in enumerate(top5):
        h = history[history['country']==ctry][['year','co2']].dropna()
        ax.plot(h['year'], h['co2'], color=pal[i], lw=2.5, label=ctry, zorder=5)
    ax.set_xlabel('Year', color=TICK, fontsize=11)
    ax.set_ylabel('CO₂ (million tonnes)', color=TICK, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
    ax.legend(fontsize=10, framealpha=0.9, facecolor=BG, edgecolor='#ddd')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("<div class='source-tag'>Source: Our World in Data · CO₂ and Greenhouse Gas Emissions dataset</div>", unsafe_allow_html=True)


#  PAGE 4 — ABOUT
elif "About" in page:

    st.markdown("""
    <div class='section-head'>
      <h3>About this Platform</h3>
      <p>Everything you need to know about the data, models, and methodology behind this dashboard.</p>
    </div>""", unsafe_allow_html=True)

    # ── Row 1: Models + Thresholds ──
    col1, col2, col3 = st.columns([1.1, 1, 1])

    with col1:
        m_status = "✅ Loaded" if models_ok else "❌ .pkl files not found"
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>ML Models</div>
          <div style='font-size:13px;color:#555;margin-bottom:4px'>{m_status}</div>
          <div style='height:1px;background:#f0ede8;margin:10px 0'></div>

          <div style='display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #f0ede8'>
            <div>
              <div style='font-size:13px;font-weight:600;color:#1a1a1a'>Decision Tree</div>
              <div style='font-size:11px;color:#888'>Interpretable baseline model</div>
            </div>
            <div style='font-family:Playfair Display,serif;font-size:20px;font-weight:700;color:#c0392b'>94.17%</div>
          </div>

          <div style='display:flex;justify-content:space-between;align-items:center;padding:8px 0'>
            <div>
              <div style='font-size:13px;font-weight:600;color:#1a1a1a'>Random Forest</div>
              <div style='font-size:11px;color:#888'>100 trees · improved accuracy</div>
            </div>
            <div style='font-family:Playfair Display,serif;font-size:20px;font-weight:700;color:#1e7c45'>95.74%</div>
          </div>

          <div style='height:1px;background:#f0ede8;margin:10px 0'></div>
          <div style='font-size:11px;color:#aaa'>Trained on 7,665 rows · 219 countries · 1990–2024</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>ML Classification Thresholds</div>
          <div style='font-size:11px;color:#888;margin-bottom:10px'>Boundaries calculated automatically from data using pd.qcut — splits dataset into equal thirds.</div>
          <div style='height:1px;background:#f0ede8;margin-bottom:12px'></div>

          <div style='display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f0ede8'>
            <span style='font-size:18px'>🟢</span>
            <div>
              <div style='font-size:13px;font-weight:600;color:#1e7c45'>Low</div>
              <div style='font-size:12px;color:#555'>0 → 2.05 Mt / year</div>
            </div>
          </div>
          <div style='display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f0ede8'>
            <span style='font-size:18px'>🟡</span>
            <div>
              <div style='font-size:13px;font-weight:600;color:#d4860a'>Medium</div>
              <div style='font-size:12px;color:#555'>2.05 → 28.76 Mt / year</div>
            </div>
          </div>
          <div style='display:flex;align-items:center;gap:10px;padding:8px 0'>
            <span style='font-size:18px'>🔴</span>
            <div>
              <div style='font-size:13px;font-weight:600;color:#c0392b'>High</div>
              <div style='font-size:12px;color:#555'>Above 28.76 Mt / year</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>IPCC Global Standard</div>
          <div style='font-size:11px;color:#888;margin-bottom:10px'>Fixed scientific thresholds based on per-capita CO₂ and the Paris Agreement carbon budget.</div>
          <div style='height:1px;background:#f0ede8;margin-bottom:12px'></div>

          <div style='display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f0ede8'>
            <span style='font-size:18px'>🟢</span>
            <div>
              <div style='font-size:13px;font-weight:600;color:#1e7c45'>Low</div>
              <div style='font-size:12px;color:#555'>Below 3t per person / year</div>
            </div>
          </div>
          <div style='display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f0ede8'>
            <span style='font-size:18px'>🟡</span>
            <div>
              <div style='font-size:13px;font-weight:600;color:#d4860a'>Medium</div>
              <div style='font-size:12px;color:#555'>3 → 10t per person / year</div>
            </div>
          </div>
          <div style='display:flex;align-items:center;gap:10px;padding:8px 0'>
            <span style='font-size:18px'>🔴</span>
            <div>
              <div style='font-size:13px;font-weight:600;color:#c0392b'>High</div>
              <div style='font-size:12px;color:#555'>Above 10t per person / year</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Row 2: Data source + methodology ──
    col4, col5 = st.columns([1, 1])

    with col4:
        st.markdown("""
        <div class='section-head'>
          <h3>Data Source</h3>
          <p>Where the data comes from</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='metric-card'>
          <div style='display:flex;align-items:flex-start;gap:14px'>
            <div style='font-size:32px'>🌍</div>
            <div>
              <div style='font-size:15px;font-weight:700;color:#1a1a1a;font-family:Playfair Display,serif;margin-bottom:4px'>Our World in Data</div>
              <div style='font-size:12px;color:#555;line-height:1.7'>
                CO₂ and Greenhouse Gas Emissions dataset.<br>
                Covers <strong>219 countries</strong> from <strong>1750 to 2024</strong>.<br>
                This dashboard uses data from <strong>1990–2024</strong> only<br>
                (earlier years have too many missing values).
              </div>
              <div style='margin-top:10px'>
                <a href='https://github.com/owid/co2-data' target='_blank'
                   style='font-size:12px;color:#c0392b;text-decoration:none;font-weight:600'>
                  → github.com/owid/co2-data
                </a>
              </div>
            </div>
          </div>
          <div style='height:1px;background:#f0ede8;margin:14px 0'></div>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;text-align:center'>
            <div>
              <div style='font-family:Playfair Display,serif;font-size:22px;font-weight:700;color:#1a1a1a'>219</div>
              <div style='font-size:11px;color:#888'>countries</div>
            </div>
            <div>
              <div style='font-family:Playfair Display,serif;font-size:22px;font-weight:700;color:#1a1a1a'>34</div>
              <div style='font-size:11px;color:#888'>years of data</div>
            </div>
            <div>
              <div style='font-family:Playfair Display,serif;font-size:22px;font-weight:700;color:#1a1a1a'>7,665</div>
              <div style='font-size:11px;color:#888'>training rows</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with col5:
      st.markdown("""
        <div class='section-head'>
          <h3>Methodology</h3>
          <p>How the models were built and validated</p>
        </div>
        
        <div class='metric-card'>
          <div class='metric-label'>Missing value handling</div>
          <p style='font-size:13px;color:#333;line-height:1.8;margin:8px 0'>
            Two-layer median fill: each country's own historical median first,
            then global median as fallback for countries with no data at all.
            Saved 4,455 extra rows vs. dropping nulls.
          </p>
        </div>

        <div class='metric-card'>
          <div class='metric-label'>Label creation</div>
          <p style='font-size:13px;color:#333;line-height:1.8;margin:8px 0'>
            pd.qcut splits total CO₂ into equal thirds — exactly 2,555 rows per class.
            Balanced classes prevent the model from being biased toward the most common label.
          </p>
        </div>

        <div class='metric-card'>
          <div class='metric-label'>Train / test split</div>
          <p style='font-size:13px;color:#333;line-height:1.8;margin:8px 0'>
            80% train+val · 20% test · stratified to preserve class ratios in both sets.
          </p>
        </div>

        <div class='metric-card'>
          <div class='metric-label'>Validation</div>
          <p style='font-size:13px;color:#333;line-height:1.8;margin:8px 0'>
            5-Fold Stratified Cross Validation on the 85% train+val set.<br>
            RF CV mean: 96.61% ± 0.76% · DT CV mean: 95.04% ± 0.74%.
          </p>
        </div>

        <div class='metric-card'>
          <div class='metric-label'>Libraries</div>
          <p style='font-size:13px;color:#333;line-height:1.8;margin:8px 0'>
            scikit-learn · pandas · numpy · matplotlib · Streamlit · Anthropic Claude API
          </p>
        </div>
      """, unsafe_allow_html=True)

    st.divider()

    # ── Row 3: Features used ──
    st.markdown("""
    <div class='section-head'>
      <h3>Features Used in the Model</h3>
      <p>The 7 input variables the ML model was trained on, ranked by importance</p>
    </div>""", unsafe_allow_html=True)

    feature_info = [
        ('oil_co2',           '#e67e22', 'Oil CO₂',           '39.8%', 'CO₂ from oil combustion (million tonnes). Strongest single predictor globally.'),
        ('gdp',               '#2980b9', 'GDP',               '17.7%', 'Gross domestic product (USD). Wealthier countries generally emit more.'),
        ('gas_co2',           '#f39c12', 'Gas CO₂',           '16.2%', 'CO₂ from natural gas combustion (million tonnes).'),
        ('coal_co2',          '#c0392b', 'Coal CO₂',          '9.8%',  'CO₂ from coal combustion. Strongest driver for High emitters in Asia.'),
        ('methane',           '#8e44ad', 'Methane',           '6.6%',  'Methane emissions from livestock, gas leaks, and landfills.'),
        ('population',        '#16a085', 'Population',        '5.9%',  'Total country population. Larger countries emit more in absolute terms.'),
        ('energy_per_capita', '#27ae60', 'Energy per capita', '4.0%',  'Total energy consumption per person (kWh). Reflects energy intensity of the economy.'),
    ]

    for feat, clr, label, importance, description in feature_info:
        imp_float = float(importance.replace('%','')) / 100
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:16px;padding:12px 16px;
                    background:#fff;border:1px solid #e8e5de;border-radius:8px;
                    margin-bottom:8px'>
          <div style='width:44px;height:44px;border-radius:8px;
                      background:{clr}22;border:1px solid {clr}66;
                      display:flex;align-items:center;justify-content:center;flex-shrink:0'>
            <div style='width:14px;height:14px;border-radius:50%;background:{clr}'></div>
          </div>
          <div style='flex:1;min-width:0'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:5px'>
              <span style='font-size:14px;font-weight:600;color:#1a1a1a'>{label}</span>
              <span style='font-size:11px;font-weight:600;color:{clr};
                           background:{clr}18;padding:2px 8px;border-radius:3px'>{importance}</span>
            </div>
            <div style='background:#f0ede8;border-radius:2px;height:6px;margin-bottom:6px'>
              <div style='width:{imp_float*100:.1f}%;height:100%;background:{clr};border-radius:2px'></div>
            </div>
            <div style='font-size:12px;color:#666'>{description}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='source-tag' style='margin-top:8px'>Feature importance from Random Forest · trained on Our World in Data CO₂ dataset</div>", unsafe_allow_html=True)