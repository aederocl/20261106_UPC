#include <stdio.h>
#include <math.h>
static double g(double x){ return exp(-0.5*x*x)/sqrt(2*M_PI); }
int main(void){
    FILE *gp = popen("gnuplot", "w");
    fprintf(gp, "set terminal pngcairo size 840,480 font 'sans,11'\nset output 'plot_c.png'\n"
        "set title 'C / gnuplot'\nset xlabel 'x'\nset ylabel 'f(x)'\nset border 3\nset xtics nomirror\nset ytics nomirror\n"
        "set key top right\n$curve << EOD\n");
    for(int i=0;i<=800;i++){ double x=-4+8.0*i/800; fprintf(gp,"%g %g\n",x,g(x)); }
    fprintf(gp,"EOD\n$area << EOD\n");
    for(int i=0;i<=200;i++){ double x=i/200.0; fprintf(gp,"%g %g\n",x,g(x)); }
    fprintf(gp,"EOD\nplot $area with filledcurves y1=0 lc rgb '#e07b39' fs transparent solid 0.6 title 'area 0→1 = 0.341345', "
               "$curve with lines lw 2 lc rgb '#1f4e79' title '1/√(2π) e^{-x²/2}'\n");
    pclose(gp); return 0;
}
