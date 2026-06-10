import streamlit as st

from src.config import MODEL_GROUPS

st.session_state["use_gpu"] = False

def render_sidebar():

    st.sidebar.title("⚙️ Configuration")

    # =====================================================
    # MODEL SELECTION
    # =====================================================
    st.sidebar.subheader("Model Selection")

    model_group = st.sidebar.selectbox(
        "Model Category",
        list(MODEL_GROUPS.keys())
    )

    model_dict = MODEL_GROUPS[model_group]

    model_name = st.sidebar.selectbox(
        "Foundation Model",
        list(model_dict.keys())
    )

    st.session_state["model_name"] = model_dict[model_name]

    # =====================================================
    # MODEL SETTINGS
    # =====================================================
    st.sidebar.subheader("Model Settings")

    st.session_state["max_seq_length"] = st.sidebar.selectbox(
        "Max Sequence Length",
        [128, 256, 512, 1024, 2048],
        index=2
    )

    st.session_state["load_in_4bit"] = st.sidebar.checkbox(
        "Use QLoRA (4-bit)",
        value=True
    )

    # =====================================================
    # LORA SETTINGS
    # =====================================================
    st.sidebar.subheader("LoRA Settings")

    st.session_state["r"] = st.sidebar.slider("Rank (r)", 4, 64, 16)
    st.session_state["lora_alpha"] = st.sidebar.slider("Alpha", 4, 64, 16)

    st.session_state["lora_dropout"] = st.sidebar.slider(
        "Dropout",
        0.0,
        0.5,
        0.05
    )

    # =====================================================
    # TRAINING SETTINGS
    # =====================================================
    st.sidebar.subheader("Training")

    st.session_state["batch_size"] = st.sidebar.number_input(
        "Batch Size",
        value=4,
        step=1,
        format="%d"
    )

    st.session_state["gradient_accumulation_steps"] = st.sidebar.number_input(
        "Gradient Accumulation",
        value=4,
        step=1,
        format="%d"
    )

    st.session_state["epochs"] = st.sidebar.slider(
        "Epochs",
        1,
        300,
        30
    )

    st.session_state["learning_rate"] = st.sidebar.number_input(
        "Learning Rate",
        value=2e-4,
        step=0.0001,
        format="%.4f"
    )

    st.session_state["optim"] = st.sidebar.selectbox(
        "Optimizer",
        ["adamw_8bit", "adamw_torch"]
    )