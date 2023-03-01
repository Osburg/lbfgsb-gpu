#include "examples/normApproximation/normApproximation_cuda.h"

#ifdef __INTELLISENSE__
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
#endif

#include <cublas_v2.h>
#include <stdio.h>

// set initial values of x
__global__ void normMin_kernel_init(int n, float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.;
}


void normMin_cuda(int& m, int& n, float* A, float* b, float* x, float& f, float* fgrad, float** assist_buffer, int task) {
  dim3 blocksize = {64U, 1U, 1U};
  dim3 gridsize = {(n + 63U) / 64U, 1U, 1U};

  //
  // Compute the standard starting point if task == 'XS'
  //
  if (task == 'XS') {
    normMin_kernel_init<<<gridsize, blocksize>>>(n, x);
    cudaMalloc(assist_buffer, m * sizeof(float));
    return;
  }

  bool feval = task == 'F' || task == 'FG';
  bool geval = task == 'G' || task == 'FG';

  //
  // Compute function value / gradient if feval or geval
  //
  if (feval || geval) {
    // create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    //
    // Do the matrix multiplication Ax - b
    //
    float alpha = 1.;
    float beta = -1.;
    float* temp = *assist_buffer;
    cudaMemcpy(temp, b, m*sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, temp, 1);

    if (feval) {
        // compute cost function ||Ax-b||^2 / m
        cublasSnrm2(handle, m, temp, 1, &f);
        f = f*f / static_cast<float>(m);
    }

    if (geval) {
        // compute gradient 2 * A.T (Ax - b) / m
        alpha = 2. / static_cast<float>(m);
        beta = 0.;
        cublasSgemv(handle, CUBLAS_OP_T, m, n, &alpha, A, m, temp, 1, &beta, fgrad, 1);
    }

    // destroy handle
    cublasDestroy(handle);

  }

}
