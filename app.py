import json
import streamlit as st

st.title("LLM Fine-Tuning Studio")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["json"]
)

epochs = st.sidebar.slider(
    "Epochs",
    1,
    50,
    20
)

batch_size = st.sidebar.slider(
    "Batch Size",
    1,
    16,
    4
)

if uploaded_file:

    data = json.load(uploaded_file)

    st.success(
        f"{len(data)} examples loaded."
    )

if st.button("Start Training"):

    st.info(
        "Training job configuration generated."
    )

    st.write(
        {
            "epochs": epochs,
            "batch_size": batch_size
        }
    )