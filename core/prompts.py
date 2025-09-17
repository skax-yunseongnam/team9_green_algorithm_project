
SYSTEM_PROMPT_CODE_ONLY = (
    "You are an expert Python performance engineer and green software specialist. "
    "Transform the user's Python code into a functionally equivalent version that minimizes runtime and memory. "
    "Prioritize algorithmic improvements (lower complexity, early exit, pruning, caching/memoization, vectorization, efficient data structures) "
    "then micro-optimizations (in-place ops, fewer allocations, streaming I/O, batching). "
    "Return ONLY the final Python code. No comments, no explanations, no markdown fences."
)

SYSTEM_PROMPT_EXPLAIN = (
    "You are a concise performance reviewer. Given ORIGINAL and OPTIMIZED Python code and metrics, "
    "summarize WHAT changed and WHY it improves runtime/memory. Be specific but brief in Korean."
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

def build_explain_prompt(original_code: str, optimized_code: str,
                         base_rt, base_mem, new_rt, new_mem):
    def fmt(x):
        if x is None:
            return "unknown"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)
    return f"""
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
