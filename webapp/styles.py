# =============================================================================
# styles.py — Light theme CSS for the Handwritten Digit Recognizer app.
#
# Design language:
#   • Clean white cards with soft shadows — feels like a modern web dashboard
#   • Blue (#2563EB) as the primary accent — trustworthy, professional
#   • Slate grey (#475569) for secondary text — readable but not harsh
#   • Generous padding and rounded corners — spacious and approachable
#   • Clear visual hierarchy: hero → section label → content card → detail
# =============================================================================


def get_css() -> str:
    """Return the CSS string injected into the Streamlit page."""
    return """
<style>
/* ── Google Font ────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Page background ────────────────────────────────────────────────────── */
.stApp {
    background: #F1F5F9;
}

/* ── Main content block ─────────────────────────────────────────────────── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   HERO BANNER
   ════════════════════════════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 50%, #1E40AF 100%);
    border-radius: 20px;
    padding: 2.8rem 2rem 2.4rem;
    text-align: center;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.25);
    position: relative;
    overflow: hidden;
}
/* subtle decorative circles */
.hero::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -50px; left: -30px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 0.5rem;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.hero p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.85);
    margin: 0 auto;
    max-width: 540px;
    position: relative;
    z-index: 1;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}

/* ════════════════════════════════════════════════════════════════════════════
   SECTION LABELS  (small caps above each panel)
   ════════════════════════════════════════════════════════════════════════════ */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}

/* ════════════════════════════════════════════════════════════════════════════
   CARDS  — white elevated containers
   ════════════════════════════════════════════════════════════════════════════ */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
    border: 1px solid #E2E8F0;
    margin-bottom: 1rem;
}

/* ════════════════════════════════════════════════════════════════════════════
   INSTRUCTION BOX  (left panel guidance)
   ════════════════════════════════════════════════════════════════════════════ */
.instructions {
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
}
.instructions p {
    color: #1E40AF;
    font-size: 0.88rem;
    margin: 0.25rem 0;
    line-height: 1.5;
}
.instructions strong { color: #1D4ED8; }

/* ════════════════════════════════════════════════════════════════════════════
   CANVAS WRAPPER
   ════════════════════════════════════════════════════════════════════════════ */
.canvas-wrapper {
    background: #FFFFFF;
    border: 2px solid #DBEAFE;
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.08), inset 0 0 0 1px #EFF6FF;
    display: inline-block;
    width: 100%;
}

/* ════════════════════════════════════════════════════════════════════════════
   PREDICTION RESULT CARD
   ════════════════════════════════════════════════════════════════════════════ */
.result-card {
    background: linear-gradient(145deg, #FFFFFF 0%, #EFF6FF 100%);
    border: 1.5px solid #BFDBFE;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(37, 99, 235, 0.10);
    margin-bottom: 1rem;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}
.result-digit {
    font-size: 7rem;
    font-weight: 800;
    color: #2563EB;
    line-height: 1;
    letter-spacing: -4px;
    text-shadow: 0 4px 24px rgba(37,99,235,0.18);
}
.result-confidence {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.6rem;
}
.result-sublabel {
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 0.3rem;
}

/* Confidence level colours */
.conf-high   { color: #16A34A; }   /* green   ≥ 90% */
.conf-medium { color: #D97706; }   /* amber   70–89% */
.conf-low    { color: #DC2626; }   /* red     < 70% */

/* ════════════════════════════════════════════════════════════════════════════
   KPI METRIC CARDS  (dashboard row)
   ════════════════════════════════════════════════════════════════════════════ */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #2563EB;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94A3B8;
    margin-top: 0.2rem;
}

/* ════════════════════════════════════════════════════════════════════════════
   EMPTY STATE MESSAGE
   ════════════════════════════════════════════════════════════════════════════ */
.empty-msg {
    text-align: center;
    padding: 3rem 1rem 2.5rem;
    color: #94A3B8;
    font-size: 0.95rem;
    line-height: 1.8;
}
.empty-msg .icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 0.6rem;
    opacity: 0.6;
}
.empty-msg strong { color: #64748B; }

/* ════════════════════════════════════════════════════════════════════════════
   DIVIDER
   ════════════════════════════════════════════════════════════════════════════ */
.divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.2rem 0;
}

/* ════════════════════════════════════════════════════════════════════════════
   TABS  — override Streamlit tab styling
   ════════════════════════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: #FFFFFF;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   BUTTONS
   ════════════════════════════════════════════════════════════════════════════ */
.stButton > button {
    width: 100%;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.52rem 0 !important;
    transition: all 0.15s ease !important;
    border: 1.5px solid transparent !important;
}
/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.30) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 5px 18px rgba(37,99,235,0.40) !important;
    transform: translateY(-1px) !important;
}
/* Secondary buttons */
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #475569 !important;
    border: 1.5px solid #E2E8F0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   STREAMLIT NATIVE OVERRIDES
   ════════════════════════════════════════════════════════════════════════════ */
/* Metrics */
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stMetricValue"] {
    color: #2563EB !important;
    font-weight: 800 !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    background: #FFFFFF !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #E2E8F0 !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
    border-radius: 999px !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
}
</style>
"""
