import os
import streamlit as st


def _read_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


API_BASE_URL = (os.getenv("API_BASE_URL") or _read_secret("API_BASE_URL", "")).rstrip("/")
API_TIMEOUT = int(os.getenv("API_TIMEOUT") or _read_secret("API_TIMEOUT", 30))
