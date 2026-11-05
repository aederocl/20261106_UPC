f(x) = exp(-0.5*x*x)/sqrt(2π)
function simpson(a, b, n)
    h = (b-a)/n
    s = f(a) + f(b)
    for i in 1:n-1
        s += f(a+i*h) * (isodd(i) ? 4.0 : 2.0)
    end
    return s*h/3
end
simpson(0.0, 1.0, 10)                 # warm-up: JIT compile
t = @elapsed I = simpson(0.0, 1.0, 100_000_000)
println("Julia   I = ", round(I, digits=15), "  time = ", round(t, digits=4), " s  (excl. JIT warm-up)")
