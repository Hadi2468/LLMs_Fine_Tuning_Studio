import json
import streamlit as st

from src.train import train_model
from src.config import DATA_PATH
from src.training_config.training_config import TrainingConfig


def render_training_panel():

    st.header("🚀 Fine-Tuning Panel")

    # -------------------------
    # Dataset upload
    # -------------------------
    uploaded_file = st.file_uploader(
        "Upload Dataset (JSON)",
        type=["json"]
    )

    if uploaded_file:

        data = json.load(uploaded_file)

        st.success(f"{len(data)} samples loaded")

        st.json(data[:2])

        save_path = DATA_PATH["data_dir"] / "train_data.json"

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        st.session_state["dataset_ready"] = True

    # -------------------------
    # Train config builder
    # -------------------------
    config = TrainingConfig(
        model_name=st.session_state.get("model_name"),
        max_seq_length=st.session_state.get("max_seq_length"),
        load_in_4bit=st.session_state.get("load_in_4bit"),

        r=st.session_state.get("r"),
        lora_alpha=st.session_state.get("lora_alpha"),
        lora_dropout=st.session_state.get("lora_dropout"),

        batch_size=st.session_state.get("batch_size"),
        gradient_accumulation_steps=st.session_state.get("gradient_accumulation_steps"),
        epochs=st.session_state.get("epochs"),
        learning_rate=st.session_state.get("learning_rate"),
        optim=st.session_state.get("optim"),
    )

    # -------------------------
    # Training button
    # -------------------------
    if st.button("🚀 Start Fine-Tuning", use_container_width=True):

        if not st.session_state.get("dataset_ready"):
            st.warning("Please upload dataset first!")
            return
        
        config.dataset_path = str(DATA_PATH["data_dir"] / "train_data.json")
        
        with st.spinner("Training in progress... (this may take a while)"):

            train_model(config)

        st.success("Training completed successfully!")

        st.session_state["model_ready"] = True