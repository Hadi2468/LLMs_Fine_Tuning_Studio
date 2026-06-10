import torch
import streamlit as st

from src.ui.sidebar import render_sidebar
from src.ui.training_panel import render_training_panel
from src.ui.inference_panel import render_inference_panel


st.set_page_config(
    page_title="LLMs Fine-Tuning Studio",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LLMs Fine-Tuning Studio")

# Sidebar (controls config)
render_sidebar()

tab1, tab2 = st.tabs(["🚀 Training", "💬 Chat Inference"])

with tab1:
    render_training_panel()

with tab2:
    render_inference_panel()

if not torch.cuda.is_available():
    st.error("🚨 No GPU detected. Training will not work with Unsloth.")

# --------------------------------------------------------------------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    **🧑🏻‍💻 Author**  
    **Hadi Hosseini**  
    AI/ML Engineer  
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Hadi2468/LLMs_Fine_Tuning_Studio)
                    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hadi468)
""")