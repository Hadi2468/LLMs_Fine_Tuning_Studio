import streamlit as st
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference import generate


@st.cache_resource
def load_model_and_tokenizer(model_dir):

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )

    model.eval()
    return model, tokenizer


def render_inference_panel():

    st.header("💬 Chat with Fine-Tuned Model")

    model_path = st.session_state.get("model_path")

    model = None
    tokenizer = None

    # -------------------------
    # Resolve correct model dir
    # -------------------------
    if model_path:

        base_dir = Path(model_path)
        merged_dir = Path(str(model_path) + "_merged")

        if merged_dir.exists():
            model_dir = merged_dir
            st.success(f"Using merged model: {model_dir.name}")

        elif base_dir.exists():
            model_dir = base_dir
            st.warning("Merged model not found → using adapter model")

        else:
            st.error("Model directory not found")
            return

        model, tokenizer = load_model_and_tokenizer(model_dir)

    # -------------------------
    # Chat history
    # -------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -------------------------
    # Input
    # -------------------------
    user_input = st.chat_input("Ask something...")

    if user_input:

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                if model is None:
                    response = "⚠️ No model loaded."
                else:
                    response = generate(user_input, model, tokenizer)

            st.write(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })