
import difflib

def make_unified_diff(original_code: str, optimized_code: str) -> str:
    diff_lines = difflib.unified_diff(
        original_code.splitlines(),
        optimized_code.splitlines(),
        fromfile="original.py",
        tofile="optimized.py",
        lineterm=""
    )
    return "\n".join(diff_lines)
