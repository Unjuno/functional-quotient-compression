#include <stddef.h>

void film_apply(const double *x, const double *g, const double *b, double *y, int n, int d) {
    for (int i=0;i<n;i++) for (int j=0;j<d;j++) y[(size_t)i*d+j]=x[(size_t)i*d+j]*g[j]+b[j];
}

void givens_fused(const double *x, const double *g, const double *b, const double *c, const double *s,
                  double *y, int n, int d) {
    for (int i=0;i<n;i++) {
        for (int j=0;j<d;j+=2) {
            int k=j/2; double u=x[(size_t)i*d+j]*g[j]+b[j];
            double v=x[(size_t)i*d+j+1]*g[j+1]+b[j+1];
            y[(size_t)i*d+j]=c[k]*u-s[k]*v;
            y[(size_t)i*d+j+1]=s[k]*u+c[k]*v;
        }
    }
}

void affine_apply(const double *x, const double *a, const double *b, double *y, int n, int d) {
    for (int i=0;i<n;i++) for (int j=0;j<d;j++) {
        double z=b[j]; for (int k=0;k<d;k++) z+=x[(size_t)i*d+k]*a[(size_t)j*d+k];
        y[(size_t)i*d+j]=z;
    }
}

void lowrank_apply(const double *x, const double *g, const double *b, const double *u, const double *v,
                   double *y, int n, int d, int r) {
    for (int i=0;i<n;i++) {
        for (int j=0;j<d;j++) y[(size_t)i*d+j]=x[(size_t)i*d+j]*g[j]+b[j];
        for (int k=0;k<r;k++) {
            double z=0.0; for (int j=0;j<d;j++) z+=x[(size_t)i*d+j]*v[(size_t)k*d+j];
            for (int j=0;j<d;j++) y[(size_t)i*d+j]+=z*u[(size_t)j*r+k];
        }
    }
}
