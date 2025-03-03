# app.py
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import importlib
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth

# Check for streamlit_authenticator
try:
    importlib.import_module('streamlit_authenticator')
except ImportError:
    st.error("Error: streamlit_authenticator is not installed. Please install it using 'pip install streamlit-authenticator'.")
    st.stop()

# Check for auth.py
if not os.path.exists("auth.py"):
    st.error("Error: auth.py file not found. Make sure it's in the same directory as app.py.")
    st.stop()

import auth  # Import the authentication logic

# Import custom functions
from Home import Home
from cluster1 import classify
from association import association
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

authenticator = auth.authenticate()  # Get the authenticator object

# --- SIDEBAR ---
menu_choice = st.sidebar.radio("Navigation", ["Login", "Sign Up", "Forgot Password", "Reset Password", "Login with Google"])

# --- PAGES ---
if menu_choice == "Login":
    if authenticator.login("Login", "main"): # Corrected login argument
        st.write(f"Welcome, {authenticator.get_current_user()}")
        # --- Main App Content ---
        st.title("Edupreneurialship Prediction Model")

        selected = option_menu(
            'Main Menu',
            ['Home', 'Classification', 'Association', 'Prediction'],
            icons=['house', 'list-check', 'shuffle', 'graph-up'],
            default_index=0,
            menu_icon="cast",
            orientation="horizontal"
        )
        if selected == "Home":
            Home()
        elif selected == "Classification":
            classify()
        elif selected == "Association":
            association()
        elif selected == "Prediction":
            predict()

        if authenticator.logout("Logout", "main"):
            st.write("Goodbye!")
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")

elif menu_choice == "Sign Up":
    try:
        if authenticator.register_user("Register user", "main"):
            st.success("User registered successfully")
            auth.save_user_data(authenticator.credentials)  # Save data using auth.py function
    except Exception as e:
        st.error(f"Error: {e}")

elif menu_choice == "Forgot Password":
    try:
        if authenticator.forgot_password("Forgot password", "main"):
            st.success("Password modified successfully")
            auth.save_user_data(authenticator.credentials)
    except Exception as e:
        st.error(f"Error: {e}")

elif menu_choice == "Reset Password":
    try:
        if authenticator.reset_password("Reset password", "main"):
            st.success("Password reset successfully")
            auth.save_user_data(authenticator.credentials)
    except Exception as e:
        st.error(f"Error: {e}")

elif menu_choice == "Login with Google":
    if st.button("Login with Google"):
        try:
            creds = auth.google_login()
            if creds:
                st.success("Google login successful!")
                st.title("Edupreneurialship Prediction Model (Google Login)")

                selected = option_menu(
                    'Main Menu',
                    ['Home', 'Classification', 'Association', 'Prediction'],
                    icons=['house', 'list-check', 'shuffle', 'graph-up'],
                    default_index=0,
                    menu_icon="cast",
                    orientation="horizontal"
                )
                if selected == "Home":
                    Home()
                elif selected == "Classification":
                    classify()
                elif selected == "Association":
                    association()
                elif selected == "Prediction":
                    predict()

        except Exception as e:
            st.error(f"Google login failed: {e}")
