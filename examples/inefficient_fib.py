
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

n = 35
print(f"{n}번째 피보나치 수:", fib(n))
