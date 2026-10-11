# =============================================================================
# app.py — Handwritten Digit Recognizer
#
# Run:  streamlit run webapp/app.py   (from digit_recognition/ folder)
#
# Layout:
#   Top bar     — dark-mode toggle
#   Hero banner — real accuracy from cached metrics
#   Tab 1 — Draw & Predict    : canvas (left) + results panel (right)
#   Tab 2 — Prediction History: session-scoped stats + table + CSV + delete
#   Tab 3 — Model Dashboard   : KPIs + per-class table + F1 chart + confusion
# =============================================================================

import os
import sys
import math
import logging
import uuid
import numpy as np
import streamlit as st
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger("digit_app")

# ── Path setup ────────────────────────────────────────────────────────────────
_WEBAPP_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_WEBAPP_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ── Local imports ─────────────────────────────────────────────────────────────
from webapp.preprocessor import (preprocess_canvas_image, segment_canvas_image,
                                 get_example_image, is_canvas_empty)
from webapp.predictor    import load_model, predict
from webapp.charts       import make_probability_chart
from webapp.styles       import get_css
from webapp.history      import (init_db, save_prediction, get_history,
                                  get_summary_stats, clear_history,
                                  delete_prediction, set_feedback,
                                  export_history_csv)
from webapp.dashboard    import (load_metrics, load_cached_metrics,
                                  render_key_metrics,
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
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE
# =============================================================================
def _init_state():
    defaults = {
        "result"        : None,        # prediction dict | "empty" | "error:…"
        "preview_img"   : None,        # PIL 28×28 thumbnail
        "example_data"  : None,        # RGBA array for canvas background
        "canvas_key"    : 0,           # increment to reset the canvas widget
        "confirm_clear" : False,       # two-step confirmation guard
        "session_id"    : uuid.uuid4().hex,  # isolates this visitor's history
        "dark_mode"     : False,       # theme toggle
        "multi_mode"    : False,       # multi-digit segmentation toggle
        "last_pred_id"  : None,        # row id of the last saved prediction
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Sidebar settings (dark toggle must be read before CSS injection) ──────────
with st.sidebar:
    st.markdown('<div class="side-brand">✏️ Digit Recognizer</div>',
                unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    st.toggle("🌙 Dark mode", key="dark_mode",
              help="Switch between light and dark theme")
    st.markdown(
        '<p class="side-note">History is scoped to your browser session — '
        'other visitors can never see your predictions.</p>',
        unsafe_allow_html=True,
    )
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ── One-time startup ──────────────────────────────────────────────────────────
init_db()

# ── Real metrics for the hero banner / footer (no evaluation triggered) ───────
_cached_metrics = load_cached_metrics()
if _cached_metrics:
    _acc_pct    = _cached_metrics["test_accuracy"] * 100
    _acc_badge  = f"CNN · MNIST · {_acc_pct:.2f} % Accuracy"
    _acc_footer = f"{_acc_pct:.2f}% test accuracy"
else:
    _acc_badge  = "CNN · MNIST"
    _acc_footer = "MNIST-trained CNN"

# ── Sidebar: live session stats ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">This session</p>',
                unsafe_allow_html=True)
    _session_stats = get_summary_stats(st.session_state.session_id)
    _s1, _s2 = st.columns(2)
    _s1.metric("Predictions", _session_stats["total"])
    _s2.metric("Avg Confidence",
               "—" if _session_stats["total"] == 0
               else f"{_session_stats['avg_conf']*100:.1f}%")
    st.markdown(
        '<p class="side-note">Prediction history is private to this '
        'browser session. Drawings are never stored.</p>',
        unsafe_allow_html=True,
    )


def _donut_svg(pct: float, size: int = 150, stroke: int = 13) -> str:
    """
    Return an inline-SVG circular confidence gauge.

    The gauge curve colours itself by confidence (green/amber/red) and the
    track/text colours follow the active theme via CSS variables.
    """
    pct    = float(np.clip(pct, 0.0, 100.0))
    r      = (size - stroke) / 2
    circ   = 2 * math.pi * r
    frac   = max(pct / 100.0, 0.001)
    colour = "#16A34A" if pct >= 90 else "#D97706" if pct >= 70 else "#DC2626"
    c      = size / 2
    return f"""
    <svg class="donut-gauge" width="{size}" height="{size}"
         viewBox="0 0 {size} {size}" role="img" aria-label="{pct:.1f}% confidence">
        <circle cx="{c:.0f}" cy="{c:.0f}" r="{r:.1f}" fill="none"
                stroke="var(--gauge-track)" stroke-width="{stroke}"/>
        <circle cx="{c:.0f}" cy="{c:.0f}" r="{r:.1f}" fill="none"
                stroke="{colour}" stroke-width="{stroke}" stroke-linecap="round"
                stroke-dasharray="{circ*frac:.2f} {circ:.2f}"
                transform="rotate(-90 {c:.0f} {c:.0f})"/>
        <text x="50%" y="50%" text-anchor="middle" dominant-baseline="central"
              class="gauge-value" fill="var(--gauge-text)">{pct:.1f}%</text>
    </svg>"""


def _conf_label(confidence: float) -> tuple:
    """Return (css_class, human_label) for a confidence value."""
    if confidence >= 0.90:
        return "conf-high", "🟢 High confidence"
    if confidence >= 0.70:
        return "conf-medium", "🟡 Medium confidence"
    return "conf-low", "🔴 Low confidence — try drawing more clearly"


# =============================================================================
# HERO BANNER
# =============================================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">✨ {badge}</div>
    <h1>✏️ Handwritten Digit Recognizer</h1>
</div>
""".format(badge=_acc_badge), unsafe_allow_html=True)


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
            <p>🔢 Turn on <strong>Multi-digit mode</strong> to recognise 2–4 digits at once.</p>
        </div>
        """, unsafe_allow_html=True)

        # Canvas
        CANVAS_SIZE  = 280
        STROKE_WIDTH = 18

        # ── Stroke width slider (must render before the canvas so its
        #    value is available when the canvas is drawn) ───────────────────
        stroke_val = st.slider(
            "✏️ Stroke width",
            min_value=8, max_value=28, value=STROKE_WIDTH, step=2,
            help="Thicker strokes generally give better results"
        )

        bg_image = None
        if st.session_state.example_data is not None:
            bg_image = Image.fromarray(
                st.session_state.example_data.astype(np.uint8), mode="RGBA"
            )

        st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color       = "#000000",
            stroke_width     = stroke_val,
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

        # ── Multi-digit toggle ─────────────────────────────────────────────
        st.toggle(
            "🔢 Multi-digit mode",
            key="multi_mode",
            help="Split the canvas into separate digits and read them left-to-right",
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

        # ── Multi-digit prediction ────────────────────────────────────────
        elif isinstance(result, dict) and result.get("multi"):
            digits = result["multi"]
            combined = "".join(str(d["digit"]) for d in digits)
            avg_conf = float(np.mean([d["confidence"] for d in digits]))
            conf_class, _ = _conf_label(avg_conf)

            pills = "".join(
                f'<div class="digit-pill">'
                f'<div class="n">{d["digit"]}</div>'
                f'<div class="c">{d["confidence"]*100:.1f}% confidence</div>'
                f'</div>'
                for d in digits
            )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-layout">
                    <div class="donut-block">{_donut_svg(avg_conf*100, size=130)}</div>
                    <div class="digit-block">
                        <div class="result-label">Recognised Number</div>
                        <div class="digit-big multi">{combined}</div>
                        <span class="conf-tag {conf_class}">
                            {len(digits)} digit(s) detected
                        </span>
                    </div>
                </div>
            </div>
            <div class="digit-grid">
                {pills}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1rem;">Per-digit detail</p>',
                        unsafe_allow_html=True)
            for i, d in enumerate(digits):
                st.markdown(
                    f"**{i+1}.** Digit **{d['digit']}** — "
                    f"{d['confidence']*100:.1f}% confidence"
                )

        # ── Successful single prediction ──────────────────────────────────
        else:
            digit      = result["digit"]
            confidence = result["confidence"]
            probs      = result["probabilities"]

            conf_class, conf_label = _conf_label(confidence)

            # Result card — confidence donut + big digit
            st.markdown(f"""
            <div class="result-card">
                <div class="result-layout">
                    <div class="donut-block">
                        {_donut_svg(confidence*100)}
                    </div>
                    <div class="digit-block">
                        <div class="result-label">Predicted Digit</div>
                        <div class="digit-big">{digit}</div>
                        <span class="conf-tag {conf_class}">{conf_label}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar chart
            st.markdown('<p class="section-label" style="margin-top:1rem;">All class probabilities</p>',
                        unsafe_allow_html=True)
            prob_fig = make_probability_chart(probs, digit)
            st.pyplot(prob_fig, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(prob_fig)

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

            # ── Feedback (correct the prediction) ─────────────────────────
            pred_id = st.session_state.last_pred_id
            if pred_id is not None:
                fb1, fb2 = st.columns([2, 1])
                with fb1:
                    actual = st.selectbox(
                        "Was this correct? Select the true digit:",
                        list(range(10)), index=int(digit),
                        key=f"feedback_{pred_id}",
                    )
                with fb2:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    if st.button("✔️ Submit feedback"):
                        try:
                            set_feedback(pred_id, actual)
                            st.session_state.last_pred_id = None
                            st.toast("Thanks! Your feedback was recorded.", icon="✅")
                            st.rerun()
                        except Exception as e:
                            logger.exception("Saving feedback failed")
                            st.error(f"Could not save feedback: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # BUTTON HANDLERS  (outside columns so st.rerun() is safe)
    # ─────────────────────────────────────────────────────────────────────────
    if clear_clicked:
        st.session_state.canvas_key   += 1
        st.session_state.example_data  = None
        st.session_state.result        = None
        st.session_state.preview_img   = None
        st.session_state.last_pred_id  = None
        st.rerun()

    if example_clicked:
        try:
            st.session_state.example_data = get_example_image(digit=example_digit)
            st.session_state.canvas_key  += 1
            st.session_state.result       = None
            st.session_state.preview_img  = None
            st.session_state.last_pred_id = None
        except Exception as e:
            logger.exception("Loading MNIST example failed")
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
            st.session_state.last_pred_id = None
        else:
            with st.spinner("Analysing your drawing …"):
                try:
                    model = load_model()
                    if st.session_state.multi_mode:
                        tensors, previews, is_empty = segment_canvas_image(image_data)
                        if is_empty:
                            st.session_state.result = "empty"
                            st.session_state.last_pred_id = None
                        else:
                            multi = [predict(model, t) for t in tensors]
                            st.session_state.result      = {"multi": multi}
                            st.session_state.preview_img = previews[0] if previews else None
                            st.session_state.last_pred_id = None
                            for res in multi:
                                save_prediction(
                                    session_id    = st.session_state.session_id,
                                    predicted     = res["digit"],
                                    confidence    = res["confidence"],
                                    probabilities = res["probabilities"],
                                )
                    else:
                        tensor, preview_img, is_empty = preprocess_canvas_image(image_data)
                        if is_empty:
                            st.session_state.result = "empty"
                            st.session_state.last_pred_id = None
                        else:
                            res = predict(model, tensor)
                            st.session_state.result      = res
                            st.session_state.preview_img = preview_img
                            st.session_state.last_pred_id = save_prediction(
                                session_id    = st.session_state.session_id,
                                predicted     = res["digit"],
                                confidence    = res["confidence"],
                                probabilities = res["probabilities"],
                            )
                except Exception as e:
                    logger.exception("Prediction failed")
                    st.session_state.result = f"error:{e}"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 📋 Prediction History")
    st.caption(
        "Every prediction is logged automatically for **this browser session only**. "
        "Only the digit, confidence, and timestamp are stored — "
        "**drawings are never saved.**"
    )

    stats = get_summary_stats(st.session_state.session_id)

    if stats["total"] > 0:
        # ── KPI row ───────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", stats["total"])
        k2.metric("Avg Confidence",    f"{stats['avg_conf']*100:.1f}%")
        k3.metric("Most Predicted",    f"Digit  {stats['most_common']}")
        k4.metric("Unique Digits",     len(stats["digit_counts"]))

        # ── Export CSV ────────────────────────────────────────────────────
        st.download_button(
            "⬇️  Download history (CSV)",
            data=export_history_csv(st.session_state.session_id),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── Two-column layout: table (wide) + distribution chart ──────────
        ht_left, ht_right = st.columns([1.6, 1.0], gap="large")

        with ht_left:
            st.markdown('<p class="section-label">Recent Predictions</p>',
                        unsafe_allow_html=True)
            rows = get_history(st.session_state.session_id, limit=100)
            if rows:
                table_data = []
                for r in rows:
                    top3_str = "  ·  ".join(
                        [f"{d}: {p*100:.1f}%" for d, p in r["top3"]]
                    )
                    pct   = r["confidence"] * 100
                    badge = "🟢" if pct >= 90 else "🟡" if pct >= 70 else "🔴"
                    fb    = "" if r["is_correct"] is None else (
                        "✅" if r["is_correct"] else f"✏️ {r['user_label']}"
                    )
                    table_data.append({
                        "#"         : r["id"],
                        "Time"      : r["timestamp"],
                        "Digit"     : r["predicted"],
                        "Confidence": f"{badge} {pct:.1f}%",
                        "Feedback"  : fb,
                        "Top 3"     : top3_str,
                    })

                import pandas as pd
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
                        "Feedback"  : st.column_config.TextColumn("Feedback", width="small"),
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
            plt.close(fig)

        # ── Delete a single record ────────────────────────────────────────
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Delete a single record</p>',
                    unsafe_allow_html=True)
        del_c1, del_c2, _ = st.columns([1.5, 1, 3])
        with del_c1:
            row_ids = [r["id"] for r in rows]
            del_id  = st.selectbox("Record #", row_ids, key="delete_row_id")
        with del_c2:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("🗑️ Delete record"):
                delete_prediction(del_id)
                st.toast(f"Deleted record #{del_id}.", icon="🗑️")
                st.rerun()

        # ── Clear history (two-step confirmation) ─────────────────────────
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        if not st.session_state.confirm_clear:
            if st.button("🧹  Clear My History", type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("⚠️ This will permanently delete **your** prediction records. Are you sure?")
            cc1, cc2, _ = st.columns([1, 1, 4])
            with cc1:
                if st.button("✅  Yes, delete all", type="primary"):
                    deleted = clear_history(st.session_state.session_id)
                    st.session_state.confirm_clear = False
                    st.toast(f"Deleted {deleted} records.", icon="🗑️")
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
            model = load_model()
            metrics = load_metrics(model)
        except Exception as e:
            logger.exception("Loading metrics failed")
            st.error(f"Could not load metrics: {e}")
            st.info("Run `python main.py` first to train the model and generate metrics.")
            st.stop()

    if st.session_state.pop("_metrics_just_updated", False):
        st.success("Metrics recomputed and updated.")

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
        from config import EVAL_METRICS_PATH
        if os.path.exists(EVAL_METRICS_PATH):
            os.remove(EVAL_METRICS_PATH)
        from webapp.dashboard import compute_and_cache_metrics
        load_metrics.clear()
        with st.spinner("Running model on 10,000 test images …"):
            compute_and_cache_metrics(model)
        st.session_state["_metrics_just_updated"] = True
        st.rerun()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.78rem; padding-bottom:1.5rem;">
    Handwritten Digit Recognition &nbsp;·&nbsp; Phase 3
    &nbsp;·&nbsp; Streamlit + TensorFlow + SQLite
    &nbsp;·&nbsp; CNN trained on MNIST (60,000 samples)
    &nbsp;·&nbsp; {acc}
</div>
""".format(acc=_acc_footer), unsafe_allow_html=True)
