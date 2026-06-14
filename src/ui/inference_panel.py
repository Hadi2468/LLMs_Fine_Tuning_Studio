import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from src.inference import generate


@st.cache_resource
def load_model_and_tokenizer(model_dir):

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
            
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )

    model.eval()

    return model, tokenizer


def render_inference_panel():

    st.header("💬 Chat with Fine-Tuned Model")

    model_path = st.session_state.get("model_path", None)

    # -------------------------
    # Load model if selected
    # -------------------------
    model = None
    tokenizer = None
  
    if model_path:

        merged_model_path = model_path + "_merged"

        if not Path(merged_model_path).exists():
            st.error(f"Merged model not found:\n{merged_model_path}")
            return

        model, tokenizer = load_model_and_tokenizer(merged_model_path)

        st.success(f"Loaded model from: {merged_model_path}")
         
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