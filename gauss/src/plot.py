import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import erf, sqrt
x = np.linspace(-4, 4, 800); g = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)
xs = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
ax.plot(x, g(x), color="#1f4e79", lw=2, label=r"$\frac{1}{\sqrt{2\pi}}e^{-x^2/2}$")
ax.fill_between(xs, g(xs), color="#e07b39", alpha=.6, label=f"area 0→1 = {0.5*erf(1/sqrt(2)):.6f}")
ax.set(xlabel="x", ylabel="f(x)", title="Python / matplotlib"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("plot_python.png")
