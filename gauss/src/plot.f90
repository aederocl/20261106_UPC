program plot
  implicit none
  real(8), parameter :: pi = 3.14159265358979323846d0
  integer :: i, u
  real(8) :: x
  open(newunit=u, file='f_curve.dat'); do i = 0, 800
     x = -4d0 + 8d0*i/800; write(u,*) x, exp(-0.5d0*x*x)/sqrt(2d0*pi); end do; close(u)
  open(newunit=u, file='f_area.dat');  do i = 0, 200
     x = i/200d0;          write(u,*) x, exp(-0.5d0*x*x)/sqrt(2d0*pi); end do; close(u)
  open(newunit=u, file='f_plot.gp')
  write(u,'(A)') "set terminal pngcairo size 840,480 font 'sans,11'"
  write(u,'(A)') "set output 'plot_fortran.png'"
  write(u,'(A)') "set title 'Fortran / gnuplot'; set xlabel 'x'; set ylabel 'f(x)'"
  write(u,'(A)') "set border 3; set xtics nomirror; set ytics nomirror"
  write(u,'(A)') "plot 'f_area.dat' with filledcurves y1=0 lc rgb '#e07b39' fs transparent solid 0.6 &
       &title 'area 0→1 = 0.341345', \"
  write(u,'(A)') "     'f_curve.dat' with lines lw 2 lc rgb '#1f4e79' title '1/√(2π) e^{-x²/2}'"
  close(u)
  call execute_command_line('gnuplot f_plot.gp')
end program
