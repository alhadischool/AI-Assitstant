import re
import streamlit as st

def apply_rtl_styles():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap" rel="stylesheet">
        <style>
            h1, h2, h3, h4, p, label, .stMarkdown, .stText, 
            [data-testid="stSidebarUserContent"], 
            [data-testid="stChatMessageContent"], 
            .stChatInput textarea {
                font-family: 'Vazirmatn', sans-serif !important;
            }

            .block-container, 
            [data-testid="stSidebarUserContent"],
            .stChatInput textarea,
            [data-testid="stChatMessageContent"],
            h1, h2, h3, p, .stMarkdown, .stCheckbox {
                direction: rtl !important;
                text-align: right !important;
            }

            p, div {
                line-height: 1.8;
            }
        </style>
    """, unsafe_allow_html=True)

def render_styled_text(text: str):
    if re.search(r'[\u0600-\u06FF]', text):
        st.markdown(f'<div style="direction: rtl; text-align: right;">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="direction: ltr; text-align: left;">{text}</div>', unsafe_allow_html=True)