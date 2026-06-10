import streamlit as st

from src.inference import generate


def render_inference_panel():

    st.header("💬 Chat with Fine-Tuned Model")

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
    # User input
    # -------------------------
    user_input = st.chat_input("Ask something...")

    if user_input:

        # show user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # generate response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = generate(user_input)

            st.write(response)

        # save assistant response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })