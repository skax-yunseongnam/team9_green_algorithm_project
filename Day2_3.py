#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit


# In[2]:


#pip install streamlit-ace


# In[1]:


#pip install openai


# In[2]:


import streamlit as st
from streamlit_ace import st_ace
import time
import tracemalloc
import os
import difflib

# =========================
# OpenAI 클라이언트 준비
# =========================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    st.warning("openai 패키지가 필요하다. 터미널에서 `pip install openai` 후 다시 실행하라.")

def get_client():
    if OpenAI is None:
        return None
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# =========================
# 실행 + 메모리 측정
# =========================
def run_code(code: str):
    tracemalloc.start()
    start_time = time.time()
    try:
        exec_globals = {}
        exec(code, exec_globals)  # ⚠️ 임의 코드 실행. 신뢰된 코드만 사용하라.
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

def request_green_optimized_code(client: "OpenAI", model: str, original_code: str,
                                 runtime_sec: float | None, memory_kb: float | None) -> str:
    prompt = build_user_prompt(original_code, runtime_sec, memory_kb)
    resp = client.chat.completions.create(
        model=model,
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
# 변경 이유 요약 (텍스트)
# =========================
SYSTEM_PROMPT_EXPLAIN = (
    "You are a concise performance reviewer. Given ORIGINAL and OPTIMIZED Python code and metrics, "
    "summarize WHAT changed and WHY it improves runtime/memory. Be specific but brief in Korean."
)

def request_change_explanation(client: "OpenAI", model: str,
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
        model=model,
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
# 발자국 계산 (에너지/탄소)
# =========================
def compute_footprint(runtime_sec: float, peak_kb: float):
    # 상수 (필요 시 사이드바에 노출 가능)
    PUE = 1.67
    PSF = 1
    n_CPUcores = 8
    CPUpower = 15.6        # W per core
    usageCPU_used = 1
    memoryPower = 0.3725   # W/GB
    carbonIntensity = 415.6  # gCO2/kWh

    mem_gb = peak_kb / (1024.0 * 1024.0)  # KB → GB

    power_core = PUE * n_CPUcores * CPUpower * usageCPU_used  # W
    power_mem  = PUE * (mem_gb * memoryPower)                 # W
    power_tot  = power_core + power_mem                       # W

    energy_core = runtime_sec * power_core * PSF / 1000.0     # kWh
    energy_mem  = runtime_sec * power_mem  * PSF / 1000.0     # kWh
    energy_tot  = runtime_sec * power_tot  * PSF / 1000.0     # kWh

    ce_core = energy_core * carbonIntensity                   # gCO2
    ce_mem  = energy_mem  * carbonIntensity                   # gCO2
    ce_tot  = energy_tot  * carbonIntensity                   # gCO2

    r = lambda x, n=4: round(x, n)
    energy_tot = r(energy_tot); ce_tot = r(ce_tot)
    ce_core = r(ce_core); ce_mem = r(ce_mem)

    if (ce_core + ce_mem) > 0:
        ce_core_per = round(ce_core / (ce_core + ce_mem) * 100.0, 2)
        ce_mem_per  = round(ce_mem  / (ce_core + ce_mem) * 100.0, 2)
    else:
        ce_core_per = ce_mem_per = 0.0

    # 등가 지표 (요청한 공식 그대로)
    tree_days   = r(ce_tot * 30 / 11000 * 12)     # days
    train_km    = r(ce_tot / 41)
    lightbulb_h = r(energy_tot * 1000 / 60)       # 60W 전구 시간
    car_km      = r(ce_tot / 251)

    return {
        "energy_kwh": energy_tot,
        "carbon_g": ce_tot,
        "ce_core_g": ce_core,
        "ce_mem_g": ce_mem,
        "ce_core_per": ce_core_per,
        "ce_mem_per": ce_mem_per,
        "tree_days": tree_days,
        "train_km": train_km,
        "lightbulb_h": lightbulb_h,
        "car_km": car_km,
    }

# =========================
# 절감치 계산/표시
# =========================
def _pct_reduction(old: float | None, new: float | None) -> float:
    if old is None or new is None:
        return 0.0
    if old <= 0:
        return 0.0
    return (old - new) / old * 100.0

def _pos(x: float | None) -> float:
    if x is None:
        return 0.0
    return x if x > 0 else 0.0

def compute_savings(base_rt, base_mem_kb, base_fp: dict, opt_rt, opt_mem_kb, opt_fp: dict):
    saved = {
        "runtime_s": (base_rt - opt_rt) if (base_rt is not None and opt_rt is not None) else None,
        "memory_kb": (base_mem_kb - opt_mem_kb) if (base_mem_kb is not None and opt_mem_kb is not None) else None,
        "energy_kwh": base_fp["energy_kwh"] - opt_fp["energy_kwh"],
        "carbon_g":   base_fp["carbon_g"]   - opt_fp["carbon_g"],
        "tree_days":  base_fp["tree_days"]  - opt_fp["tree_days"],
        "train_km":   base_fp["train_km"]   - opt_fp["train_km"],
        "lightbulb_h":base_fp["lightbulb_h"]- opt_fp["lightbulb_h"],
        "car_km":     base_fp["car_km"]     - opt_fp["car_km"],
    }
    pct = {
        "runtime": _pct_reduction(base_rt, opt_rt),
        "memory":  _pct_reduction(base_mem_kb, opt_mem_kb),
        "energy":  _pct_reduction(base_fp["energy_kwh"], opt_fp["energy_kwh"]),
        "carbon":  _pct_reduction(base_fp["carbon_g"],   opt_fp["carbon_g"]),
    }
    return saved, pct

def show_savings_right_only(base_rt, base_mem, base_fp: dict | None,
                            opt_rt, opt_mem, opt_fp: dict):
    if not base_fp:
        st.info("원본 측정값이 없어 절감치를 계산할 수 없다. 먼저 상단의 ‘코드 실행’으로 원본을 측정하라.")
        return

    saved, pct = compute_savings(base_rt, base_mem, base_fp, opt_rt, opt_mem, opt_fp)

    left, right = st.columns([2, 1])
    with left:
        st.write("### 🌿 절감 요약 (원본 대비)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("⏱️ 실행 시간 절감", f"{_pos(saved['runtime_s']):.4f} s", f"{pct['runtime']:+.1f}%")
            st.metric("🧠 피크 메모리 절감", f"{_pos(saved['memory_kb']):.2f} KB", f"{pct['memory']:+.1f}%")
        with c2:
            st.metric("⚡ 에너지 절감", f"{_pos(saved['energy_kwh']):.4f} kWh", f"{pct['energy']:+.1f}%")
            st.metric("🌍 탄소배출 절감", f"{_pos(saved['carbon_g']):.2f} gCO₂", f"{pct['carbon']:+.1f}%")

        # 증가 시 경고
        bad = []
        if saved["energy_kwh"] < 0: bad.append("에너지")
        if saved["carbon_g"]   < 0: bad.append("탄소")
        if saved["runtime_s"]  is not None and saved["runtime_s"] < 0: bad.append("실행 시간")
        if saved["memory_kb"]  is not None and saved["memory_kb"] < 0: bad.append("메모리")
        if bad:
            st.warning("다음 항목은 오히려 증가했다: " + ", ".join(bad))

        st.write("#### 🚉 등가 지표 절감")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("🌳 나무", f"{_pos(saved['tree_days']):.2f} days")
        d2.metric("🚆 기차", f"{_pos(saved['train_km']):.2f} km")
        d3.metric("💡 전구(60W)", f"{_pos(saved['lightbulb_h']):.2f} h")
        d4.metric("🚗 자동차", f"{_pos(saved['car_km']):.2f} km")

        with st.expander("세부 수치 (원본 → 최적화)"):
            st.write(
                f"- 실행 시간: {base_rt:.4f}s → {opt_rt:.4f}s "
                f"({_pos(saved['runtime_s']):.4f}s 절감, {pct['runtime']:+.1f}%)"
            )
            st.write(
                f"- 피크 메모리: {base_mem:.2f}KB → {opt_mem:.2f}KB "
                f"({_pos(saved['memory_kb']):.2f}KB 절감, {pct['memory']:+.1f}%)"
            )
            st.write(
                f"- 에너지: {base_fp['energy_kwh']:.4f}kWh → {opt_fp['energy_kwh']:.4f}kWh "
                f"({_pos(saved['energy_kwh']):.4f}kWh 절감, {pct['energy']:+.1f}%)"
            )
            st.write(
                f"- 탄소: {base_fp['carbon_g']:.2f}g → {opt_fp['carbon_g']:.2f}g "
                f"({_pos(saved['carbon_g']):.2f}g 절감, {pct['carbon']:+.1f}%)"
            )

    with right:
        st.write("### 🖼️ 인포 카드")
        st.image("https://img.icons8.com/fluency/96/000000/deciduous-tree.png", caption="나무 흡수(절감)")
        st.image("https://img.icons8.com/fluency/96/000000/train.png", caption="기차 주행(절감)")
        st.image("https://img.icons8.com/fluency/96/000000/light-on.png", caption="60W 전구(절감)")
        st.image("https://img.icons8.com/fluency/96/000000/car.png", caption="자동차 주행(절감)")
        st.caption("아이콘은 예시이다. 로컬/사내 가이드 아이콘으로 교체 가능하다.")

# =========================
# UI
# =========================
st.set_page_config(page_title="그린 알고리즘 코드 실행기", layout="wide")
st.title("Python 코드 실행기 + 그린 최적화")

# 에디터
code = st_ace(
    language="python",
    theme="monokai",
    keybinding="vscode",
    font_size=14,
    height=300,
    auto_update=True,
)

# 원본 코드 실행
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
    default_model = "gpt-4.1-mini"
    model_name = st.text_input("최적화 모델명", value=default_model, help="OpenAI 모델명을 입력하라 (예: gpt-4.1, gpt-4.1-mini 등)")

st.divider()

# 최적화 → 자동 실행 → Diff/이유/절감
if st.button("최적화 코드 생성 및 자동 실행 (코드만 반환)"):
    if not code or not code.strip():
        st.warning("먼저 상단 편집기에 코드를 입력하라.")
    else:
        client = get_client()
        if client is None:
            st.error("OpenAI 클라이언트를 초기화하지 못했다. API 키를 secrets.toml 또는 환경변수로 설정하라.")
        else:
            base_rt = st.session_state.get("last_runtime")
            base_mem = st.session_state.get("last_memory")

            with st.spinner("그린 최적화 코드를 생성 중…"):
                try:
                    optimized = request_green_optimized_code(
                        client=client,
                        model=(model_name.strip() or default_model),
                        original_code=code,
                        runtime_sec=base_rt,
                        memory_kb=base_mem,
                    )
                except Exception as e:
                    optimized = ""
                    st.error(f"최적화 요청 중 오류: {e}")

            if optimized:
                st.write("#### 추천 코드 (GPT가 반환한 코드만)")
                st.code(optimized, language="python")

                # 자동 실행
                out2, rt2, mem2 = run_code(optimized)
                st.write("#### 추천 코드 실행 결과")
                st.write(out2)
                st.write(f"실행 시간: {rt2:.4f} 초")
                st.write(f"메모리 사용량: {mem2:.2f} KB")

                # Diff
                st.write("#### 변경된 코드 Diff")
                diff_text = make_unified_diff(code or "", optimized)
                if diff_text.strip():
                    st.code(diff_text, language="diff")
                else:
                    st.write("코드 변경점이 감지되지 않았다.")

                # 변경 이유 요약
                try:
                    explanation = request_change_explanation(
                        client=client,
                        model=(model_name.strip() or default_model),
                        original_code=code or "",
                        optimized_code=optimized,
                        base_rt=base_rt, base_mem=base_mem,
                        new_rt=rt2, new_mem=mem2
                    )
                    st.write("#### 변경 요약 및 이유")
                    st.markdown(explanation)
                except Exception as e:
                    st.warning(f"변경 이유 요약 생성 중 문제: {e}")

                # 발자국 + 절감
                if base_rt is not None and base_mem is not None:
                    base_fp = compute_footprint(base_rt, base_mem)
                    opt_fp  = compute_footprint(rt2, mem2)
                    show_savings_right_only(base_rt, base_mem, base_fp, rt2, mem2, opt_fp)
                else:
                    st.info("절감치 표시를 위해 먼저 ‘코드 실행’으로 원본을 측정하라.")

                # 다운로드
                st.download_button(
                    label="최적화 코드 다운로드",
                    data=optimized.encode("utf-8"),
                    file_name="optimized.py",
                    mime="text/x-python",
                )

# ========== 예시 비효율 코드(원하면 붙여서 테스트) ==========
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


# In[ ]:




