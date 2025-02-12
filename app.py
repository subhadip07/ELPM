import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
from streamlit_option_menu import option_menu

# Import custom functions
from Home import Home
from Home import load_sidebar
from cluster import classify
from association import association
# from viz import visualization
from prediction import predict

# Set page configuration
st.set_page_config(
    page_title="Edupreneurialship Prediction Model",
    page_icon=":rocket:",
    layout="wide"
)

# Initialize session state for the DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar menu
with st.sidebar:
    # Title with centered alignment
    st.markdown("<h2 style='text-align: center;'>Edupreneurialship Prediction Model</h2>", unsafe_allow_html=True)

    # Subtitle with centered alignment
    st.markdown("<h4 style='text-align: center;'>Navigate through the sections:</h4>", unsafe_allow_html=True)
    
    selected = option_menu(
        'Main Menu',
        ['Home', 'Visualisation', 'Classification','Association', 'Prediction'],
        icons=['house', 'bar-chart-line', 'list-check', 'shuffle', 'graph-up'],
        default_index=0,
        menu_icon="cast"
    )

# Load respective page based on user selection
if selected == "Home":
    Home()
elif selected == "Classification":
    classify()
elif selected == "Association":
    association()
elif selected == "Prediction":
    predict()