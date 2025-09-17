
import os
import streamlit as st
from streamlit_ace import st_ace

from core.settings import load_settings
from core.runner import run_code
from core.aoai_client import get_client
from core.optimizer import request_green_optimized_code, request_change_explanation
from core.diff_utils import make_unified_diff
from core.footprint import compute_footprint
from ui.components import show_savings_right_only

st.set_page_config(page_title="그린 알고리즘 코드 실행기", layout="wide")
st.title("Python 코드 실행기 + 그린 최적화")

cfg = load_settings()

code = st_ace(
    language="python",
    theme="monokai",
    keybinding="vscode",
    font_size=14,
    height=300,
    auto_update=True,
)

col_run, col_model = st.columns([1, 2])
with col_run:
    if st.button("코드 실행"):
        output, runtime, memory = run_code(code or "")
        st.write("### 실행 결과")
        st.write(output)
        st.write(f"### 실행 시간: {runtime:.4f} 초")
        st.write(f"### 메모리 사용량: {memory:.2f} KB")
        st.session_state["last_runtime"] = runtime
        st.session_state["last_memory"] = memory

with col_model:
    default_deployment = os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini")
    deployment_name = st.text_input(
        "최적화 배포명",
        value=default_deployment,
        help="Azure OpenAI에 등록한 배포명 (예: gpt-4o, gpt-4o-mini 등)"
    )

st.divider()

if st.button("최적화 코드 생성 및 자동 실행 (코드만 반환)"):
    if not code or not code.strip():
        st.warning("먼저 상단 편집기에 코드를 입력하라.")
    else:
        client = get_client()
        if client is None:
            st.error("Azure OpenAI 클라이언트를 초기화하지 못했다.")
        else:
            base_rt = st.session_state.get("last_runtime")
            base_mem = st.session_state.get("last_memory")

            with st.spinner("그린 최적화 코드를 생성 중…"):
                try:
                    optimized = request_green_optimized_code(
                        client=client,
                        deployment=(deployment_name.strip() or default_deployment),
                        original_code=code,
                        runtime_sec=base_rt,
                        memory_kb=base_mem,
                    )
                except Exception as e:
                    optimized = ""
                    st.error(f"최적화 요청 중 오류: {e}")

            if optimized:
                st.write("#### 추천 코드")
                st.code(optimized, language="python")

                out2, rt2, mem2 = run_code(optimized)
                st.write("#### 추천 코드 실행 결과")
                st.write(out2)
                st.write(f"실행 시간: {rt2:.4f} 초")
                st.write(f"메모리 사용량: {mem2:.2f} KB")

                st.write("#### 변경된 코드 Diff")
                diff_text = make_unified_diff(code or "", optimized)
                if diff_text.strip():
                    st.code(diff_text, language="diff")
                else:
                    st.write("코드 변경점이 감지되지 않았다.")

                try:
                    explanation = request_change_explanation(
                        client=client,
                        deployment=(deployment_name.strip() or default_deployment),
                        original_code=code or "",
                        optimized_code=optimized,
                        base_rt=base_rt, base_mem=base_mem,
                        new_rt=rt2, new_mem=mem2
                    )
                    st.write("#### 변경 요약 및 이유")
                    st.markdown(explanation)
                except Exception as e:
                    st.warning(f"변경 이유 요약 생성 중 문제: {e}")

                if base_rt is not None and base_mem is not None:
                    base_fp = compute_footprint(base_rt, base_mem)
                    opt_fp = compute_footprint(rt2, mem2)
                    show_savings_right_only(base_rt, base_mem, base_fp, rt2, mem2, opt_fp)
                else:
                    st.info("절감치 표시를 위해 먼저 ‘코드 실행’으로 원본을 측정하라.")

                st.download_button(
                    label="최적화 코드 다운로드",
                    data=optimized.encode("utf-8"),
                    file_name="optimized.py",
                    mime="text/x-python",
                )

with st.expander("예시: 비효율 피보나치 코드 삽입"):
    st.code(
        'def fib(n):\n'
        '    if n <= 1:\n'
        '        return n\n'
        '    return fib(n-1) + fib(n-2)\n\n'
        'n = 35\n'
        'print(f"{n}번째 피보나치 수:", fib(n))\n',
        language="python"
    )
