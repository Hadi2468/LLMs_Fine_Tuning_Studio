import streamlit as st
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference import generate


BASE_MODEL = "meta-llama/Llama-3.2-1B"  # adjust if needed


def render_inference_panel():

    st.header("💬 Chat with Fine-Tuned Model")

    model_path = st.session_state.get("model_path", None)

    # -------------------------
    # Load model if selected
    # -------------------------
    model = None
    tokenizer = None

    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        st.success(f"Loaded model from: {model_path}")

    # -------------------------
    # Init chat history
    # -------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -------------------------
    # Show history
    # -------------------------
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
                    response = "⚠️ No model loaded. Select a training job first."
                else:
                    response = generate(user_input, model, tokenizer)

            st.write(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })