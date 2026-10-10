# =============================================================================
# app.py — Handwritten Digit Recognizer · Light Theme UI
#
# Run:  streamlit run webapp/app.py   (from digit_recognition/ folder)
#
# Layout:
#   Hero banner
#   Tab 1 — Draw & Predict    : canvas (left) + results panel (right)
#   Tab 2 — Prediction History: stats + table + digit distribution chart
#   Tab 3 — Model Dashboard   : KPIs + per-class table + F1 chart + confusion
# =============================================================================

import os
import sys
import numpy as np
import streamlit as st
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Path setup ────────────────────────────────────────────────────────────────
_WEBAPP_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_WEBAPP_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ── Local imports ─────────────────────────────────────────────────────────────
from webapp.preprocessor import preprocess_canvas_image, get_example_image, is_canvas_empty
from webapp.predictor    import load_model, predict
from webapp.charts       import make_probability_chart
from webapp.styles       import get_css
from webapp.history      import (init_db, save_prediction, get_history,
                                  get_summary_stats, clear_history)
from webapp.dashboard    import (load_metrics, render_key_metrics,
                                  render_per_class_table, render_f1_bar_chart,
                                  render_confusion_matrix, render_training_curves)
from streamlit_drawable_canvas import st_canvas


# =============================================================================
# PAGE CONFIG  — must be the very first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(get_css(), unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def _init_state():
    defaults = {
        "result"        : None,    # prediction dict | "empty" | "error:…"
        "preview_img"   : None,    # PIL 28×28 thumbnail
        "example_data"  : None,    # RGBA array for canvas background
        "canvas_key"    : 0,       # increment to reset the canvas widget
        "confirm_clear" : False,   # two-step confirmation guard
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── One-time startup ──────────────────────────────────────────────────────────
init_db()
model = load_model()


# =============================================================================
# HERO BANNER
# =============================================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">✨ CNN · MNIST · 99.5 % Accuracy</div>
    <h1>✏️ Handwritten Digit Recognizer</h1>
    <p>Draw any digit from 0 to 9, press <strong>Predict Digit</strong>,
       and watch the neural network identify it in milliseconds.</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================
tab_draw, tab_history, tab_dashboard = st.tabs([
    "🖊  Draw & Predict",
    "📋  Prediction History",
    "📊  Model Dashboard",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DRAW & PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_draw:
    st.markdown("<br/>", unsafe_allow_html=True)
    left_col, gap_col, right_col = st.columns([1.05, 0.05, 0.9])

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT — Canvas panel
    # ══════════════════════════════════════════════════════════════════════════
    with left_col:
        st.markdown('<p class="section-label">Drawing Canvas</p>',
                    unsafe_allow_html=True)

        # Instructions
        st.markdown("""
        <div class="instructions">
            <p>🖱️ <strong>Click and drag</strong> to draw a single digit (0–9).</p>
            <p>📏 Draw large and centred — filling most of the canvas gives the best accuracy.</p>
            <p>⬛ The black canvas with white stroke matches the MNIST training format exactly.</p>
        </div>
        """, unsafe_allow_html=True)

        # Canvas
        CANVAS_SIZE  = 280
        STROKE_WIDTH = 18

        bg_image = None
        if st.session_state.example_data is not None:
            bg_image = Image.fromarray(
                st.session_state.example_data.astype(np.uint8), mode="RGBA"
            )

        st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color       = "#000000",
            stroke_width     = STROKE_WIDTH,
            stroke_color     = "#FFFFFF",
            background_color = "#000000",   # black bg = MNIST convention
            background_image = bg_image,
            height           = CANVAS_SIZE,
            width            = CANVAS_SIZE,
            drawing_mode     = "freedraw",
            key              = f"canvas_{st.session_state.canvas_key}",
            display_toolbar  = False,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Stroke width slider ────────────────────────────────────────────
        stroke_val = st.slider(
            "✏️ Stroke width",
            min_value=8, max_value=28, value=STROKE_WIDTH, step=2,
            help="Thicker strokes generally give better results"
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Primary action ────────────────────────────────────────────────
        predict_clicked = st.button(
            "🔮  Predict Digit",
            type="primary",
            help="Run the CNN model on your drawing",
        )

        # ── Secondary actions ─────────────────────────────────────────────
        b1, b2 = st.columns(2)
        with b1:
            clear_clicked = st.button("🗑️  Clear Canvas", help="Erase drawing")
        with b2:
            reset_clicked = st.button("↺  Reset Results", help="Clear result panel")

        # ── Example loader ────────────────────────────────────────────────
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Load MNIST Example</p>',
                    unsafe_allow_html=True)
        ex1, ex2 = st.columns([1, 1])
        with ex1:
            example_digit = st.selectbox(
                "Pick digit", list(range(10)), index=3,
                label_visibility="collapsed",
                help="Select a digit class to load from MNIST test set",
            )
        with ex2:
            example_clicked = st.button(
                "🎲  Try Example",
                help="Load a real MNIST test-set image onto the canvas",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT — Results panel
    # ══════════════════════════════════════════════════════════════════════════
    with right_col:
        st.markdown('<p class="section-label">Prediction Results</p>',
                    unsafe_allow_html=True)

        result = st.session_state.result

        # ── Empty state ───────────────────────────────────────────────────
        if result is None:
            st.markdown("""
            <div class="empty-msg">
                <span class="icon">🤖</span>
                <strong>No prediction yet</strong><br/>
                Draw a digit on the left, then press<br/>
                <strong>Predict Digit</strong> to see results here.
            </div>
            """, unsafe_allow_html=True)

        # ── Canvas was empty ──────────────────────────────────────────────
        elif result == "empty":
            st.markdown("""
            <div class="empty-msg">
                <span class="icon">✏️</span>
                <strong>Canvas is empty</strong><br/>
                Please draw a digit before predicting.
            </div>
            """, unsafe_allow_html=True)

        # ── Error ─────────────────────────────────────────────────────────
        elif isinstance(result, str) and result.startswith("error:"):
            st.error(f"⚠️ Prediction failed: {result.replace('error:','')}")
            st.info("Try clearing the canvas and drawing again.")

        # ── Successful prediction ─────────────────────────────────────────
        else:
            digit      = result["digit"]
            confidence = result["confidence"]
            probs      = result["probabilities"]

            # Confidence level
            if confidence >= 0.90:
                conf_class = "conf-high"
                conf_label = "🟢 High confidence"
            elif confidence >= 0.70:
                conf_class = "conf-medium"
                conf_label = "🟡 Medium confidence"
            else:
                conf_class = "conf-low"
                conf_label = "🔴 Low confidence — try drawing more clearly"

            # Result card
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Digit</div>
                <div class="result-digit">{digit}</div>
                <div class="result-confidence">
                    <span class="{conf_class}">{confidence*100:.1f}%</span>
                </div>
                <div class="result-sublabel">{conf_label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar chart
            st.markdown('<p class="section-label" style="margin-top:1rem;">All class probabilities</p>',
                        unsafe_allow_html=True)
            st.pyplot(make_probability_chart(probs, digit), use_container_width=True)

            # 28×28 input preview
            if st.session_state.preview_img is not None:
                with st.expander("🔍 What the model saw (28×28 input)"):
                    preview = st.session_state.preview_img.resize(
                        (140, 140), Image.NEAREST
                    )
                    ca, cb, cc = st.columns([1, 1, 1])
                    with cb:
                        st.image(preview, caption="CNN input tensor", width=140)
                    st.caption(
                        "White digit · black background · 28×28 pixels · "
                        "float32 values in [0, 1] — exactly matches MNIST training format."
                    )

            # Full probability table
            with st.expander("📋 Full probability table"):
                col_d, col_p = st.columns([1, 2])
                col_d.markdown("**Digit**")
                col_p.markdown("**Probability**")
                for i, p in enumerate(probs):
                    c1, c2 = st.columns([1, 2])
                    c1.write(f"{'**' if i == digit else ''}  {i}  {'  ✅**' if i == digit else ''}")
                    c2.progress(float(p), text=f"{p*100:.2f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # BUTTON HANDLERS  (outside columns so st.rerun() is safe)
    # ─────────────────────────────────────────────────────────────────────────
    if clear_clicked:
        st.session_state.canvas_key   += 1
        st.session_state.example_data  = None
        st.session_state.result        = None
        st.session_state.preview_img   = None
        st.rerun()

    if example_clicked:
        try:
            st.session_state.example_data = get_example_image(digit=example_digit)
            st.session_state.canvas_key  += 1
            st.session_state.result       = None
            st.session_state.preview_img  = None
        except Exception as e:
            st.error(f"Could not load example: {e}")
        st.rerun()

    if reset_clicked:
        st.session_state.result      = None
        st.session_state.preview_img = None
        st.rerun()

    if predict_clicked:
        image_data = canvas_result.image_data
        if image_data is None or is_canvas_empty(image_data):
            st.session_state.result      = "empty"
            st.session_state.preview_img = None
        else:
            with st.spinner("Analysing your drawing …"):
                try:
                    tensor, preview_img, is_empty = preprocess_canvas_image(image_data)
                    if is_empty:
                        st.session_state.result = "empty"
                    else:
                        res = predict(model, tensor)
                        st.session_state.result      = res
                        st.session_state.preview_img = preview_img
                        save_prediction(
                            predicted     = res["digit"],
                            confidence    = res["confidence"],
                            probabilities = res["probabilities"],
                        )
                except Exception as e:
                    st.session_state.result = f"error:{e}"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 📋 Prediction History")
    st.caption(
        "Every prediction is logged automatically. "
        "Only the digit, confidence, and timestamp are stored — "
        "**drawings are never saved.**"
    )

    stats = get_summary_stats()

    if stats["total"] > 0:
        # ── KPI row ───────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", stats["total"])
        k2.metric("Avg Confidence",    f"{stats['avg_conf']*100:.1f}%")
        k3.metric("Most Predicted",    f"Digit  {stats['most_common']}")
        k4.metric("Unique Digits",     len(stats["digit_counts"]))

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── Two-column layout: table (wide) + distribution chart ──────────
        ht_left, ht_right = st.columns([1.6, 1.0], gap="large")

        with ht_left:
            st.markdown('<p class="section-label">Recent Predictions</p>',
                        unsafe_allow_html=True)
            import pandas as pd
            rows = get_history(limit=100)
            if rows:
                table_data = []
                for r in rows:
                    top3_str = "  ·  ".join(
                        [f"{d}: {p*100:.1f}%" for d, p in r["top3"]]
                    )
                    pct   = r["confidence"] * 100
                    badge = "🟢" if pct >= 90 else "🟡" if pct >= 70 else "🔴"
                    table_data.append({
                        "#"         : r["id"],
                        "Time"      : r["timestamp"],
                        "Digit"     : r["predicted"],
                        "Confidence": f"{badge} {pct:.1f}%",
                        "Top 3"     : top3_str,
                    })

                df = pd.DataFrame(table_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                    column_config={
                        "#"         : st.column_config.NumberColumn("#",    width="small"),
                        "Time"      : st.column_config.TextColumn("Time",   width="medium"),
                        "Digit"     : st.column_config.NumberColumn("Digit",width="small"),
                        "Confidence": st.column_config.TextColumn("Conf",   width="small"),
                        "Top 3"     : st.column_config.TextColumn("Top 3 Classes", width="large"),
                    }
                )

        with ht_right:
            st.markdown('<p class="section-label">Predictions per digit</p>',
                        unsafe_allow_html=True)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4.5, 4))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#F8FAFC")
            digits = list(range(10))
            counts = [stats["digit_counts"].get(d, 0) for d in digits]
            bars   = ax.bar(digits, counts, color="#2563EB", width=0.6,
                            edgecolor="#FFFFFF", linewidth=1.0, zorder=3)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            str(cnt), ha="center", va="bottom",
                            fontsize=8, color="#1E293B", fontweight="600")
            ax.set_xticks(digits)
            ax.set_xticklabels([str(d) for d in digits], color="#1E293B", fontsize=10)
            ax.set_ylabel("Count", color="#64748B", fontsize=9)
            ax.tick_params(colors="#64748B", length=3)
            ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8, linestyle="--", zorder=0)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_edgecolor("#E2E8F0")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)

        # ── Clear history (two-step confirmation) ─────────────────────────
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        if not st.session_state.confirm_clear:
            if st.button("🗑️  Clear All History", type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("⚠️ This will permanently delete **all** prediction records. Are you sure?")
            cc1, cc2, _ = st.columns([1, 1, 4])
            with cc1:
                if st.button("✅  Yes, delete all", type="primary"):
                    deleted = clear_history()
                    st.session_state.confirm_clear = False
                    st.success(f"Deleted {deleted} records.")
                    st.rerun()
            with cc2:
                if st.button("❌  Cancel"):
                    st.session_state.confirm_clear = False
                    st.rerun()

    else:
        st.markdown("""
        <div class="empty-msg">
            <span class="icon">📋</span>
            <strong>No predictions yet</strong><br/>
            Go to the <strong>Draw &amp; Predict</strong> tab and
            make some predictions — they'll appear here automatically.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance Dashboard")
    st.caption(
        "All metrics are computed by running the CNN on the **MNIST test set "
        "(10,000 images)**. These are offline evaluation results — not live "
        "user predictions. Numbers are cached in `data/eval_metrics.json`."
    )

    with st.spinner("Loading evaluation metrics …"):
        try:
            metrics = load_metrics(model)
        except Exception as e:
            st.error(f"Could not load metrics: {e}")
            st.info("Run `python main.py` first to train the model and generate metrics.")
            st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────
    render_key_metrics(metrics)
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Per-class table + F1 chart ────────────────────────────────────────
    db_col, ch_col = st.columns([1, 1.3], gap="large")
    with db_col:
        st.markdown('<p class="section-label">Per-class Precision / Recall / F1</p>',
                    unsafe_allow_html=True)
        render_per_class_table(metrics)
    with ch_col:
        st.markdown('<p class="section-label">F1-Score per digit class</p>',
                    unsafe_allow_html=True)
        render_f1_bar_chart(metrics)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Confusion matrix ──────────────────────────────────────────────────
    st.markdown('<p class="section-label">Confusion Matrix — MNIST test set (10,000 images)</p>',
                unsafe_allow_html=True)
    st.caption(
        "Row = true digit · Column = predicted digit · "
        "Diagonal = correct · Off-diagonal = errors"
    )
    render_confusion_matrix(metrics)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Training curves ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Training History (Phase 1)</p>',
                unsafe_allow_html=True)
    render_training_curves()

    # ── Recompute button ──────────────────────────────────────────────────
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    if st.button("🔄  Recompute Metrics",
                 help="Re-run evaluation on the full MNIST test set"):
        import os as _os
        _mp = _os.path.join(_PROJECT_DIR, "data", "eval_metrics.json")
        if _os.path.exists(_mp):
            _os.remove(_mp)
        from webapp.dashboard import compute_and_cache_metrics
        load_metrics.clear()
        with st.spinner("Running model on 10,000 test images …"):
            compute_and_cache_metrics(model)
        st.success("Metrics updated. Refresh the page to see new values.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.78rem; padding-bottom:1.5rem;">
    Handwritten Digit Recognition &nbsp;·&nbsp; Phase 3
    &nbsp;·&nbsp; Streamlit + TensorFlow + SQLite
    &nbsp;·&nbsp; CNN trained on MNIST (60,000 samples)
    &nbsp;·&nbsp; 99.53% test accuracy
</div>
""", unsafe_allow_html=True)
