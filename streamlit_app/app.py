import streamlit as st
from styles import get_custom_css

st.set_page_config(
    page_title="Pnömoni Tespit Sistemi",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
        <h1>🫁 Pnömoni Tespit Sistemi</h1>
        <p>
            Sol menüden proje özeti, model performansı ve canlı tahmin sayfalarına geçebilirsiniz.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)