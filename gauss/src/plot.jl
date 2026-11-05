using Plots
gr()
g(x) = exp(-x^2/2)/sqrt(2π)
x = range(-4, 4, length=800); xs = range(0, 1, length=200)
using SpecialFunctions
A = round(erf(1/sqrt(2))/2, digits=6)   # analytic value
plot(x, g.(x), lw=2, color=:royalblue4, label="1/√(2π) e^(-x²/2)", xlabel="x", ylabel="f(x)",
     title="Julia / Plots.jl", framestyle=:semi, size=(840,480))
plot!(xs, g.(xs), fillrange=0, fillalpha=.6, fillcolor=:darkorange, lw=0, label="area 0→1 = $A")
savefig("plot_julia.png")
