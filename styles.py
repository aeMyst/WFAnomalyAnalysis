import streamlit as st
from theme import BG, CARD_BG, NAVY, TEXT_DARK, TEXT_MID, TEXT_SOFT


def apply_global_styles():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BG};
        }}

        header[data-testid="stHeader"] {{
            background-color: {NAVY} !important;
        }}

        header[data-testid="stHeader"] * {{
            color: white !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {NAVY};
            width: 260px !important;
        }}

        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        /* Sidebar label */
        section[data-testid="stSidebar"] label {{
            color: white !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        /* Dropdown container */
        section[data-testid="stSidebar"] div[data-baseweb="select"] {{
            background-color: rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
            border: 1.5px solid rgba(255,255,255,0.4) !important;
        }}

        /* Dropdown inner box */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: transparent !important;
            color: white !important;
            border: none !important;
        }}

        /* Selected text */
        section[data-testid="stSidebar"] div[data-baseweb="select"] span,
        section[data-testid="stSidebar"] div[data-baseweb="select"] div {{
            color: white !important;
        }}

        /* Dropdown arrow icon */
        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
            fill: white !important;
            opacity: 1 !important;
        }}

        /* Dropdown open — options list */
        ul[role="listbox"] {{
            background-color: white !important;
            border-radius: 8px !important;
        }}

        ul[role="listbox"] li {{
            color: {TEXT_DARK} !important;
            background-color: white !important;
        }}

        ul[role="listbox"] li:hover {{
            background-color: #f0f4ff !important;
        }}

        .block-container {{
            padding-top: 4.2rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }}

        .dashboard-title {{
            font-size: 2rem;
            font-weight: 800;
            color: {NAVY};
            margin-top: 0.75rem;
            margin-bottom: 0.25rem;
        }}

        .dashboard-subtitle {{
            font-size: 1rem;
            color: {TEXT_MID};
            margin-bottom: 1.6rem;
        }}

        .kpi-card {{
            background: {CARD_BG};
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            border-top: 12px solid {NAVY};
            margin-bottom: 0.75rem;
        }}

        .kpi-label {{
            font-size: 0.85rem;
            font-weight: 700;
            color: {TEXT_MID};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.5rem;
        }}

        .kpi-value {{
            font-size: 2rem;
            font-weight: 800;
            color: {TEXT_DARK};
            line-height: 1.1;
        }}

        .kpi-sub {{
            margin-top: 0.4rem;
            font-size: 0.9rem;
            color: {TEXT_SOFT};
        }}

        .note-card {{
            background: {CARD_BG};
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            border-top: 12px solid {NAVY};
            margin-top: 0.75rem;
        }}

        div[data-testid="stPlotlyChart"] {{
            background: {CARD_BG};
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            border-top: 12px solid {NAVY};
            margin-bottom: 1rem;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 16px !important;
        }}

        div[data-baseweb="select"] {{
            background-color: white !important;
            border-radius: 8px !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {{
            color: {TEXT_DARK} !important;
        }}

        label {{
            color: {TEXT_DARK} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )