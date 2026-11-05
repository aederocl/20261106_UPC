import numpy as np, time
f = lambda x: np.exp(-0.5*x*x)/np.sqrt(2*np.pi)
n = 100_000_000
t0 = time.perf_counter()
h = 1.0/n
i = np.arange(1, n)
s = f(0.0) + f(1.0) + np.sum(f(i*h) * np.where(i & 1, 4.0, 2.0))
I = s*h/3
print(f"NumPy   I = {I:.15f}  time = {time.perf_counter()-t0:.4f} s")
