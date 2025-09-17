
from core.prompts import SYSTEM_PROMPT_CODE_ONLY, SYSTEM_PROMPT_EXPLAIN, build_user_prompt, build_explain_prompt

def _strip_md_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if "\n" in t:
            head, rest = t.split("\n", 1)
            if head.strip().lower() in ("python",):
                return rest.strip()
    return t

def request_green_optimized_code(client, deployment: str,
                                 original_code: str,
                                 runtime_sec: float | None, memory_kb: float | None) -> str:
    prompt = build_user_prompt(original_code, runtime_sec, memory_kb)
    resp = client.chat.completions.create(
        model=deployment,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_CODE_ONLY},
            {"role": "user", "content": prompt},
        ],
    )
    return _strip_md_fence(resp.choices[0].message.content)

def request_change_explanation(client, deployment: str,
                               original_code: str, optimized_code: str,
                               base_rt: float | None, base_mem: float | None,
                               new_rt: float | None, new_mem: float | None) -> str:
    user_prompt = build_explain_prompt(original_code, optimized_code, base_rt, base_mem, new_rt, new_mem)
    resp = client.chat.completions.create(
        model=deployment,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EXPLAIN},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
