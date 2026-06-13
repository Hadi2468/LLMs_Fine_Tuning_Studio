import torch
import streamlit as st
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.ui.sidebar import render_sidebar
from src.ui.training_panel import render_training_panel
from src.ui.inference_panel import render_inference_panel


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="LLMs Fine-Tuning Studio",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LLMs Fine-Tuning Studio")

# -----------------------------
# Sidebar
# -----------------------------
render_sidebar()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🚀 Training",
    "📊 Training Results",
    "💬 Chat Inference"
])


# =============================
# TAB 1 — TRAINING
# =============================
with tab1:
    render_training_panel()


# =============================
# TAB 2 — TRAINING RESULTS
# =============================
with tab2:

    st.subheader("📊 Training Runs Explorer")

    LOG_DIR = Path(r"G:\My Drive\LLMs_studio\logs")

    if not LOG_DIR.exists():
        st.error("Logs directory not found.")
        st.stop()

    jobs = sorted([p.name for p in LOG_DIR.glob("job_*") if p.is_dir()])

    if not jobs:
        st.warning("No training runs found yet.")
        st.stop()

    # store selected job globally
    st.session_state["selected_job"] = st.selectbox(
        "Select Training Job",
        jobs
    )

    selected_job = st.session_state["selected_job"]

    metrics_file = LOG_DIR / selected_job / "train_metrics.json"

    st.divider()

    # -----------------------------
    # Load metrics safely
    # -----------------------------
    if metrics_file.exists():

        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)

            metrics = data.get("metrics", {})

            st.success(f"Loaded metrics for {selected_job}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Loss", metrics.get("loss", "N/A"))
            col2.metric("Learning Rate", metrics.get("learning_rate", "N/A"))
            col3.metric("Epoch", metrics.get("epoch", "N/A"))

            st.divider()
            st.subheader("📄 Full JSON")
            st.json(data)

            # -----------------------------
            # Loss curve
            # -----------------------------
            history = metrics.get("history", [])

            if history:
                st.subheader("📈 Loss Curve")

                losses = [x.get("loss") for x in history if x.get("loss") is not None]
                steps = list(range(len(losses)))

                fig, ax = plt.subplots()
                ax.plot(steps, losses)
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss Curve")

                st.pyplot(fig)

            else:
                st.info("No training history available for plotting.")

        except Exception as e:
            st.error(f"Failed to read metrics file: {e}")

    else:
        st.warning("No metrics file found for this job.")


# =============================
# TAB 3 — INFERENCE
# =============================
with tab3:

    selected_job = st.session_state.get("selected_job", None)

    if not selected_job:
        st.warning("⚠️ Please select a training job in the Training Results tab first.")
        st.stop()

    model_path = str(
        Path(r"G:\My Drive\LLMs_studio\models") / selected_job
    )

    if st.button("🚀 Load selected model for chat"):
        st.session_state["model_path"] = model_path
        st.success(f"Model loaded: {selected_job}")

    # Render chat UI
    render_inference_panel()


# -----------------------------
# UI styling
# -----------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
**🧑🏻‍💻 Author**  
**Hadi Hosseini**  
AI/ML Engineer  
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Hadi2468/LLMs_Fine_Tuning_Studio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hadi468)
""")