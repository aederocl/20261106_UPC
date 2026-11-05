program integral
  implicit none
  integer(8), parameter :: n = 100000000_8
  real(8), parameter :: a = 0d0, b = 1d0, pi = 3.14159265358979323846d0
  real(8) :: h, s, I, t0, t1
  integer(8) :: k, c0, c1, rate
  h = (b-a)/n
  call system_clock(c0, rate)
  s = f(a) + f(b)
  do k = 1, n-1
     if (mod(k,2_8) == 1) then
        s = s + 4d0*f(a+k*h)
     else
        s = s + 2d0*f(a+k*h)
     end if
  end do
  I = s*h/3d0
  call system_clock(c1)
  print '(A,F18.15,A,F8.4,A)', 'Fortran I = ', I, '  time = ', real(c1-c0,8)/rate, ' s'
contains
  pure function f(x)
    real(8), intent(in) :: x
    real(8) :: f
    f = exp(-0.5d0*x*x)/sqrt(2d0*pi)
  end function
end program
