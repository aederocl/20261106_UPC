#include <stdio.h>
#include <math.h>
#include <time.h>
static double f(double x){ return exp(-0.5*x*x)/sqrt(2.0*M_PI); }
int main(void){
    const long n = 100000000L;               /* must be even */
    const double a = 0.0, b = 1.0, h = (b-a)/n;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double s = f(a) + f(b);
    for(long i=1;i<n;i++) s += f(a+i*h) * ((i&1)?4.0:2.0);
    double I = s*h/3.0;
    clock_gettime(CLOCK_MONOTONIC,&t1);
    printf("C       I = %.15f  time = %.4f s\n", I, (t1.tv_sec-t0.tv_sec)+1e-9*(t1.tv_nsec-t0.tv_nsec));
    return 0;
}
