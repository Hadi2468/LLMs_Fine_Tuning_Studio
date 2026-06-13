import streamlit as st
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.ui.sidebar import render_sidebar
from src.ui.training_panel import render_training_panel
from src.ui.inference_panel import render_inference_panel
from src.config import GOOGLE_DRIVE_PATH


# --------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# --------------------------------------------------
st.set_page_config(
    page_title="LLMs Fine-Tuning Studio",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# GLOBAL PATHS
# --------------------------------------------------
LOG_DIR = GOOGLE_DRIVE_PATH["logs"]
MODEL_DIR = GOOGLE_DRIVE_PATH["models"]
JOB_DIR = GOOGLE_DRIVE_PATH["jobs"]


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Navigation buttons */
div[role="radiogroup"] label {
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
}

/* General font size */
html, body, [class*="css"] {
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🧠 LLMs Fine-Tuning Studio")


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
page = st.radio(
    "",
    [
        "🚀 Training",
        "📊 Training Results",
        "💬 Chat Inference"
    ],
    horizontal=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
render_sidebar()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**🧑🏻‍💻 Author**  
**Hadi Hosseini**  
AI/ML Engineer
""")


# ==================================================
# PAGE 1 — TRAINING
# ==================================================
if page == "🚀 Training":

    render_training_panel()


# ==================================================
# PAGE 2 — TRAINING RESULTS
# ==================================================
elif page == "📊 Training Results":

    st.subheader("📊 Training Results")

    if not LOG_DIR.exists():
        st.error(f"Logs directory not found:\n{LOG_DIR}")
        st.stop()

    jobs = sorted(
        [p.name for p in LOG_DIR.glob("job_*") if p.is_dir()],
        reverse=True
    )

    if not jobs:
        st.warning("No training runs found.")
        st.stop()

    selected_job = st.selectbox(
        "Select Training Job",
        jobs
    )

    st.session_state["selected_job"] = selected_job

    metrics_file = LOG_DIR / selected_job / "train_metrics.json"

    st.write(f"📁 Metrics File: `{metrics_file}`")

    if metrics_file.exists():

        try:

            with open(metrics_file, "r") as f:
                data = json.load(f)

            metrics = data.get("metrics", {})

            st.success(f"Loaded metrics for {selected_job}")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Final Loss",
                metrics.get("final_loss", "N/A")
            )

            col2.metric(
                "Min Loss",
                metrics.get("min_loss", "N/A")
            )

            col3.metric(
                "Logged Steps",
                metrics.get("num_logs", "N/A")
            )

            # --------------------------------
            # LOSS CURVE
            # --------------------------------
            history = metrics.get("history", [])

            if history:

                losses = [
                    x.get("loss")
                    for x in history
                    if x.get("loss") is not None
                ]

                if losses:

                    st.subheader("📈 Training Loss Curve")

                    fig, ax = plt.subplots()

                    ax.plot(range(len(losses)), losses)

                    ax.set_xlabel("Log Step")
                    ax.set_ylabel("Loss")
                    ax.set_title("Training Loss")

                    st.pyplot(fig)

            st.divider()

            with st.expander("📄 View Raw JSON"):
                st.json(data)

        except Exception as e:
            st.error(f"Failed to load metrics:\n{e}")

    else:
        st.warning(
            f"No train_metrics.json found for {selected_job}"
        )


# ==================================================
# PAGE 3 — CHAT INFERENCE
# ==================================================
elif page == "💬 Chat Inference":

    selected_job = st.session_state.get("selected_job")

    if not selected_job:

        st.warning(
            "Please select a training job first from "
            "'Training Results'."
        )

    else:

        model_path = MODEL_DIR / selected_job

        st.write(f"📁 Model Path: `{model_path}`")

        if st.button("🚀 Load Selected Model"):

            st.session_state["model_path"] = str(model_path)

            st.success(
                f"Loaded model: {selected_job}"
            )

        st.divider()

        render_inference_panel()