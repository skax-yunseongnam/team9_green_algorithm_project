
import time
import tracemalloc

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
