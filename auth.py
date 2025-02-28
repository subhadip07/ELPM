# auth.py
import streamlit as st
import streamlit_authenticator as stauth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
import json

# --- CONFIGURATION ---
SECRET_KEY = "your_very_long_and_random_secret_key"  # Replace with a secure key!
CREDENTIALS_FILE = "credentials.json"  # For Google OAuth
TOKEN_PICKLE_FILE = "token.pickle"  # For Google OAuth
USER_DATA_FILE = "user_data.json"  # For local username and password

# --- UTILITY FUNCTIONS ---
def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"usernames": {}}

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def google_login():
    creds = None
    if os.path.exists(TOKEN_PICKLE_FILE):
        with open(TOKEN_PICKLE_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, ['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'])
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PICKLE_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def authenticate():
    user_data = load_user_data()
    credentials = user_data if "usernames" in user_data else {"usernames": {}}

    authenticator = stauth.Authenticate(
        credentials, "My Streamlit App", "cookie", key="unique_key", cookie_expiry_days=30
    )
    return authenticator

print("auth.py module loaded")