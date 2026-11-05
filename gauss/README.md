# Integrating a Gaussian in Fortran, C, Python and Julia

Compute and plot

    I = ∫₀¹ 1/√(2π) · exp(−x²/2) dx = ½ erf(1/√2) = 0.341344746068543

and compare four languages on the same algorithm: composite Simpson's rule, N = 10⁸ intervals.

## Results (best of 3, shared 4-core cloud VM, so timings are noisy)

| Language                      | Result            | Time     |
|-------------------------------|-------------------|----------|
| Fortran (`gfortran -O2`)      | 0.341344746068522 | 0.81 s   |
| C (`gcc -O2`)                 | 0.341344746068522 | 0.93 s   |
| Julia 1.13 (`-O3`, no JIT)    | 0.341344746068522 | 1.09 s   |
| Python + NumPy (vectorised)   | 0.341344746068543 | 2.45 s   |
| Pure Python                   | 0.341344746068522 | 32.7 s   |

C, Fortran and Julia are within run-to-run noise of each other. The ~2e-14 error of the loops is
floating-point round-off from naive summation; NumPy's pairwise sum avoids it.

Plot generation (wall time): C/gnuplot ~0.06 s, Fortran/gnuplot ~0.06 s, Python/matplotlib ~1 s,
Julia/Plots.jl ~3-4 s (mostly startup and package loading).

## Run it

Requires gcc, gfortran, gnuplot, Python 3 (numpy, matplotlib) and Julia (Plots, SpecialFunctions).

    julia -e 'using Pkg; Pkg.add(["Plots","SpecialFunctions"])'
    pip install numpy matplotlib
    make bench     # timings
    make plots     # writes figures/plot_{c,fortran,python,julia}.png

## Layout

- `src/`      integral.{c,f90,py,jl}, integral_np.py, plot.{c,f90,py,jl}
- `figures/`  the four plots
- `slides/`   seminar slides (PDF)
