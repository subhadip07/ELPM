import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import os
from streamlit_option_menu import option_menu

def load_sidebar():
    with st.sidebar:
        st.divider()

def Home():
    load_sidebar()
    st.divider()
    tabs = st.tabs(["Home"])
    with tabs[0]:
        st.header("Welcome to Edupreneurialship Prediction Model")
        st.divider()
    
    if 'df' not in st.session_state:
        st.session_state["df"] = None
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(st.session_state["df"])
        option = st.selectbox("Select an option:", ["Show dataset dimensions", "Display data description", "Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data"])

        if option == "Show dataset dimensions":
            st.write("Dataset dimensions:", st.session_state["df"].shape)
        elif option == "Display data description":
            st.write("Data description:", st.session_state["df"].describe())
        elif option == "Verify data integrity":
            st.write("Missing values in each column:", st.session_state["df"].isnull().sum())
        elif option == "Summarize numerical data statistics":
            st.write("Numerical data statistics:", st.session_state["df"].describe(include=[np.number]))
        elif option == "Summarize categorical data":
            st.write("Categorical data summary:", st.session_state["df"].describe(include=[np.object_]))
        
