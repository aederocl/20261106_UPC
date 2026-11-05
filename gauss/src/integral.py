import math, time
def f(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
def simpson(a, b, n):
    h = (b-a)/n
    s = f(a) + f(b)
    for i in range(1, n):
        s += f(a+i*h) * (4.0 if i & 1 else 2.0)
    return s*h/3
t0 = time.perf_counter()
I = simpson(0.0, 1.0, 100_000_000)
print(f"Python  I = {I:.15f}  time = {time.perf_counter()-t0:.4f} s")
