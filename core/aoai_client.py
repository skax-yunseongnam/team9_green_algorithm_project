import os
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_client():
    try:
        api_key = os.getenv("AOAI_API_KEY")
        endpoint = os.getenv("AOAI_ENDPOINT")

        if not api_key or not endpoint:
            st.error("환경 변수에서 AOAI_API_KEY 또는 AOAI_ENDPOINT를 찾을 수 없습니다.")
            return None

        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",  # 필요시 최신으로 업데이트
            azure_endpoint=endpoint,
        )
        return client
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 초기화 실패: {e}")
        return None
