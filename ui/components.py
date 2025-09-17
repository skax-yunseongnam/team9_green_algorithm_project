
import streamlit as st
from core.footprint import compute_savings, _pos

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
        st.image("assets/tree.png", caption="나무 흡수(절감)")
        st.image("assets/train.png", caption="기차 주행(절감)")
        st.image("assets/bulb.png", caption="60W 전구(절감)")
        st.image("assets/car.png", caption="자동차 주행(절감)")
        st.caption("아이콘은 예시이다. 로컬/사내 가이드 아이콘으로 교체 가능하다.")
