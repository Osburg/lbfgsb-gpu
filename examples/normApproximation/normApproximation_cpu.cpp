#include "examples/normApproximation/normApproximation_cpu.h"

#include <algorithm>
#include <cstring>

template <typename real>
void normMin_cpu(int& m, int& n, real* A, real* b, real* x, real& f, real* fgrad, real** assist_buffer, int task) {
    //
    //     Compute the standard starting point if task = 'XS'.
    //
    if (task == 'XS') {
        for (int i = 0; i < n; i++) x[i] = 1.;
        *assist_buffer = new real[m];
        return;
    }


    bool feval = task == 'F' || task == 'FG';
    bool geval = task == 'G' || task == 'FG';

    //
    // Compute function value / gradient if feval or geval
    //
    if (feval || geval) {
    //
    // Do the matrix multiplication Ax - b
    //
    real* temp = *assist_buffer;
    for (int i=0; i<m; i++) temp[i] = 0.;

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            int k = j * n + i;
            temp[j] += A[k] * x[i];
        }
        temp[j] -= b[j];
    }

    //
    //     Compute the function if task = 'F', the gradient if task = 'G', or
    //     both if task = 'FG'.
    //
    if (feval) {
        // compute cost function ||Ax - b||² / m
        f = 0.;
        for (int j = 0; j<m; j++) {
            f += temp[j] * temp[j];
        }
        f /= static_cast<float>(m);
    }

    if (geval) {
        memset(fgrad, 0., n * sizeof(real));

        // compute gradient 2 * A.T (Ax - b) / m
        for (int j=0; j<m; j++) {
            for (int i=0; i<n; i++) {
                int k = j * n + i ;
                fgrad[i] += 2 * A[k] * temp[j];
            }
        }

        for (int i=0; i<n; i++) fgrad[i] /= static_cast<float>(m);
    }

    }
}

#define INST_HELPER(real)                                                    \
  template void normMin_cpu<real>(int& m, int& n, real* A, real* b, real* x, real& f, real* fgrad, real** assist_buffer, int task);

INST_HELPER(float);
INST_HELPER(double);
