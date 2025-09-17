import streamlit as st
from streamlit_ace import st_ace
import time
import tracemalloc
import os
import difflib
from dotenv import load_dotenv
from openai import AzureOpenAI  # ✅ Azure 전용 클라이언트

# =========================
# .env 로드 & Azure OpenAI 클라이언트
# =========================
load_dotenv()

def get_client():
    try:
        client = AzureOpenAI(
            api_key=os.getenv("AOAI_API_KEY"),
            api_version="2024-08-01-preview",  # ⚠️ 필요시 최신으로 변경
            azure_endpoint=os.getenv("AOAI_ENDPOINT")
        )
        return client
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 초기화 실패: {e}")
        return None

# =========================
# 코드 실행 + 메모리 측정
# =========================
def run_code(code: str):
    tracemalloc.start()
    start_time = time.time()
    try:
        exec_globals = {}
        exec(code, exec_globals)
        output = "코드 실행 완료"
    except Exception as e:
        output = f"오류 발생: {e}"
    runtime = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return output, runtime, peak / 1024.0  # KB

# =========================
# 프롬프트 (코드-only)
# =========================
SYSTEM_PROMPT_CODE_ONLY = (
    "You are an expert Python performance engineer and green software specialist. "
    "Transform the user's Python code into a functionally equivalent version that minimizes runtime and memory. "
    "Prioritize algorithmic improvements (lower complexity, early exit, pruning, caching/memoization, vectorization, efficient data structures) "
    "then micro-optimizations (in-place ops, fewer allocations, streaming I/O, batching). "
    "Return ONLY the final Python code. No comments, no explanations, no markdown fences."
)

def build_user_prompt(original_code: str, runtime_sec: float | None, memory_kb: float | None):
    rt = f"{runtime_sec:.6f}" if runtime_sec is not None else "unknown"
    mem = f"{memory_kb:.2f}" if memory_kb is not None else "unknown"
    return f"""
[GOAL]
- Reduce runtime and memory usage while preserving functionality and I/O behaviors.
- Keep the code self-contained and runnable as-is (no external files).
- Use only Python stdlib unless strictly necessary.

[CURRENT_METRICS]
- runtime_seconds: {rt}
- peak_memory_kb: {mem}

[CONSTRAINTS]
- Maintain same inputs/outputs and side effects unless provably redundant.
- Avoid unnecessary dependencies.
- Prefer clear, maintainable code over opaque tricks.

[FORMAT]
- Output optimized Python code only.
- No backticks. No explanations. No comments.

[ORIGINAL_CODE]
{original_code}
""".strip()

# =========================
# Azure OpenAI에 최적화 요청
# =========================
def request_green_optimized_code(client: "AzureOpenAI", deployment: str,
                                 original_code: str,
                                 runtime_sec: float | None, memory_kb: float | None) -> str:
    prompt = build_user_prompt(original_code, runtime_sec, memory_kb)
    resp = client.chat.completions.create(
        model=deployment,  # ✅ Azure는 모델 대신 배포명 사용
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_CODE_ONLY},
            {"role": "user", "content": prompt},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            head, rest = text.split("\n", 1)
            if head.strip().lower() in ("python",):
                text = rest
    return text.strip()

# =========================
# 변경 이유 요약
# =========================
SYSTEM_PROMPT_EXPLAIN = (
    "You are a concise performance reviewer. Given ORIGINAL and OPTIMIZED Python code and metrics, "
    "summarize WHAT changed and WHY it improves runtime/memory. Be specific but brief in Korean."
)

def request_change_explanation(client: "AzureOpenAI", deployment: str,
                               original_code: str, optimized_code: str,
                               base_rt: float | None, base_mem: float | None,
                               new_rt: float | None, new_mem: float | None) -> str:
    def fmt(x):
        if x is None:
            return "unknown"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)
    user_prompt = f"""
[컨텍스트]
- Original runtime(s): {fmt(base_rt)}, Original peak memory(KB): {fmt(base_mem)}
- Optimized runtime(s): {fmt(new_rt)}, Optimized peak memory(KB): {fmt(new_mem)}

[요청]
- 구체적인 변경점 불릿 목록 5~10개 내외 (예: 재귀 → 반복, 메모이제이션 추가, 내장함수로 벡터화, 불필요 복사 제거 등)
- 각 항목에 성능/메모리 개선 이유를 한 줄로 설명
- 한국어. 코드블록/백틱 금지. 간결하게.

[ORIGINAL]
{original_code}

[OPTIMIZED]
{optimized_code}
""".strip()
    resp = client.chat.completions.create(
        model=deployment,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EXPLAIN},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

# =========================
# Diff 생성
# =========================
def make_unified_diff(original_code: str, optimized_code: str) -> str:
    diff_lines = difflib.unified_diff(
        original_code.splitlines(),
        optimized_code.splitlines(),
        fromfile="original.py",
        tofile="optimized.py",
        lineterm=""
    )
    return "\n".join(diff_lines)

# =========================
# UI
# =========================
st.set_page_config(page_title="그린 알고리즘 코드 실행기", layout="wide")
st.title("Python 코드 실행기 + 그린 최적화")

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
    deployment_name = st.text_input("최적화 배포명", value=default_deployment,
        help="Azure OpenAI에 등록한 배포명 (예: gpt-4o, gpt-4o-mini 등)")

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
                st.code(diff_text, language="diff")

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

                st.download_button(
                    label="최적화 코드 다운로드",
                    data=optimized.encode("utf-8"),
                    file_name="optimized.py",
                    mime="text/x-python",
                )
