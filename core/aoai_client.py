
import os
import streamlit as st
from openai import AzureOpenAI

def get_client():
    try:
        api_key = st.secrets.get("AOAI_API_KEY") or os.getenv("AOAI_API_KEY")
        endpoint = st.secrets.get("AOAI_ENDPOINT") or os.getenv("AOAI_ENDPOINT")
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=endpoint,
        )
        return client
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 초기화 실패: {e}")
        return None
