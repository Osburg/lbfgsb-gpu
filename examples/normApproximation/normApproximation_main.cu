#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdlib.h>

#include "culbfgsb/culbfgsb.h"
#include "examples/normApproximation/normApproximation_cpu.h"
#include "examples/normApproximation/normApproximation_cuda.h"


template <typename real>
void init_cpu(real** A, real** b, int& m, int& n, int& nbd, real& xl, real& xu, int size, int lowerBound) {
    m = size;
    n = size;
    nbd = 0;
    if (lowerBound == 1) nbd = 1;
    xl = 0.;
    xu = std::numeric_limits<real>::max();
    *A = new real[m*n];
    *b = new real[m];

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            (*A)[i*n+j] = static_cast<float>((i+j) % 100) / 100.;
        }
    }
    for (int i=0; i<m; i++) (*b)[i] = static_cast<float>(i % 100) / 10.;

    return;
}

template <typename real>
void init_cuda(real** A, real** b, int& m, int& n, int& nbd, real& xl, real& xu, int size, int lowerBound) {
    m = size;
    n = size;
    nbd = 0;
    if (lowerBound == 1) nbd = 1;
    xl = 0.;
    xu = std::numeric_limits<real>::max();
    *A = new real[m * n];
    *b = new real[m];

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            (*A)[j*m+i] = static_cast<float>((i+j) % 100) / 100.;
        }
    }
    for (int i = 0; i < m; i++) (*b)[i] = static_cast<float>(i % 100) / 10.;

    return;
}

// test CPU mode
template <typename real>
real test_normMin_cpu(int size, bool lowerBound) {
  // initialize LBFGSB option
  LBFGSB_CUDA_OPTION<real> lbfgsb_options;

  lbfgsbcuda::lbfgsbdefaultoption<real>(lbfgsb_options);
  lbfgsb_options.mode = LCM_NO_ACCELERATION;
  lbfgsb_options.eps_f = static_cast<real>(1e-8);
  lbfgsb_options.eps_g = static_cast<real>(1e-8);
  lbfgsb_options.eps_x = static_cast<real>(1e-8);
  lbfgsb_options.machine_epsilon = static_cast<real>(1e-8);
  lbfgsb_options.max_iteration = 1000;
  lbfgsb_options.hessian_approximate_dimension = 8;

  // initialize LBFGSB state
  LBFGSB_CUDA_STATE<real> state;
  memset(&state, 0, sizeof(state));
  real* assist_buffer_cpu = nullptr;

  // declare and initialize A, b, m, n, nbd, xu, xl
  int m = 0;
  int n = 0;
  int nbd = 0;
  real xl = 0.;
  real xu = 0.;
  real* A = nullptr;
  real* b = nullptr;
  init_cpu(&A, &b,m, n, nbd, xl, xu, size, lowerBound);

  real minimal_f = std::numeric_limits<real>::max();
  // setup callback function that evaluate function value and its gradient
  state.m_funcgrad_callback = [&assist_buffer_cpu, &minimal_f, &m, &n, &A, &b](
                                  real* x, real& f, real* g,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<real>& summary) {
    normMin_cpu<real>(m,n,A,b,x,f,g, &assist_buffer_cpu, 'FG');
    std::cout << "CPU iteration " << summary.num_iteration << " F: " << f << std::endl;

    minimal_f = fmin(minimal_f, f);
    return 0;
  };

  // initialize CPU buffers
  int N_elements = n;

  real* x = new real[N_elements];
  real* g = new real[N_elements];

  real* _xl = new real[N_elements];
  real* _xu = new real[N_elements];

  // set boundaries
  for (int i=0; i<N_elements; i++) {
      _xl[i] = xl;
      _xu[i] = xu;
  }

  // initialize starting point
  real f_init = std::numeric_limits<real>::max();
  normMin_cpu<real>(m,n,A,b,x,f_init,g, &assist_buffer_cpu, 'XS');

  // initialize number of bounds (0 for this example)
  int* _nbd = new int[N_elements];
  for (int i=0; i<N_elements; i++) _nbd[i] = nbd;

  LBFGSB_CUDA_SUMMARY<real> summary;
  memset(&summary, 0, sizeof(summary));

  // call optimization
  auto start_time = std::chrono::steady_clock::now();
  lbfgsbcuda::lbfgsbminimize<real>(N_elements, state, lbfgsb_options, x, _nbd,
                                   _xl, _xu, summary);
  auto end_time = std::chrono::steady_clock::now();
  std::cout << std::endl << "Summary: " << std::endl;
  std::cout << "Total time: "
            << (std::chrono::duration<float, std::milli>(end_time - start_time)
                        .count() /
                static_cast<float>(1))
            << " ms" << std::endl;
  std::cout << "Iteration time: "
            << (std::chrono::duration<real, std::milli>(end_time - start_time)
                    .count() /
                static_cast<real>(summary.num_iteration))
            << " ms / iteration" << std::endl;
  std::cout << "residual f: " << static_cast<float>(summary.residual_f) << std::endl;
  std::cout << "residual g: " << static_cast<float>(summary.residual_g) << std::endl;
  std::cout << "residual x: " << static_cast<float>(summary.residual_x) << std::endl;
  std::cout << "iterations: " << static_cast<float>(summary.num_iteration) << std::endl;
  std::cout << "info: " << static_cast<float>(summary.info) << std::endl;

  // release allocated memory
  delete[] x;
  delete[] g;
  delete[] _xl;
  delete[] _xu;
  delete[] _nbd;
  delete[] assist_buffer_cpu;
  delete[] A;
  delete[] b;

  return minimal_f;
}

// test CUDA mode
float test_normMin_cuda(int size, int lowerBound) {
    // initialize LBFGSB option
    LBFGSB_CUDA_OPTION<float> lbfgsb_options;

    lbfgsbcuda::lbfgsbdefaultoption<float>(lbfgsb_options);
    lbfgsb_options.mode = LCM_CUDA;
    lbfgsb_options.eps_f = static_cast<float>(1e-8);
    lbfgsb_options.eps_g = static_cast<float>(1e-8);
    lbfgsb_options.eps_x = static_cast<float>(1e-8);
    lbfgsb_options.machine_epsilon = static_cast<float>(1e-8);
    lbfgsb_options.max_iteration = 2000;
    lbfgsb_options.hessian_approximate_dimension = 8;

    // initialize LBFGSB state
    LBFGSB_CUDA_STATE<float> state;
    memset(&state, 0, sizeof(state));
    float* assist_buffer_cuda = nullptr;
    cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));
    if (CUBLAS_STATUS_SUCCESS != stat) {
        std::cout << "CUBLAS init failed (" << stat << ")" << std::endl;
        exit(0);
    }

    // declare and initialize A, b, m, n, nbd, xu, xl
    int m = 0;
    int n = 0;
    int nbd = 0;
    float xl = 0.;
    float xu = 0.;
    float* A = nullptr;
    float* b = nullptr;

    float* A_h = nullptr;
    float* b_h = nullptr;
    init_cuda(&A_h, &b_h, m, n, nbd, xl, xu, size, lowerBound);

    cudaMalloc(&A, m*n*sizeof(A[0]));
    cudaMalloc(&b, m*sizeof(b[0]));
    cudaMemcpy(A, A_h, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_h, m*sizeof(float), cudaMemcpyHostToDevice);

    float minimal_f = std::numeric_limits<float>::max();
    // setup callback function that evaluate function value and its gradient
    state.m_funcgrad_callback = [&assist_buffer_cuda, &minimal_f, &m, &n, &A, &b](
            float* x, float& f, float* g,
            const cudaStream_t& stream,
            const LBFGSB_CUDA_SUMMARY<float>& summary) {
      normMin_cuda(m,n,A,b,x,f,g,&assist_buffer_cuda,'FG');
      std::cout << "CUDA iteration " << summary.num_iteration << " F: " << f << std::endl;

      minimal_f = fmin(minimal_f, f);
      return 0;
    };

    // initialize CUDA buffers
    int N_elements = n;

    float* x = nullptr;
    float* g = nullptr;

    float* _xl = nullptr;
    float* _xu = nullptr;
    int* _nbd = nullptr;

    cudaMalloc(&x, N_elements * sizeof(x[0]));
    cudaMalloc(&g, N_elements * sizeof(g[0]));

    cudaMalloc(&_xl, N_elements * sizeof(_xl[0]));
    cudaMalloc(&_xu, N_elements * sizeof(_xu[0]));

    // set boundaries
    float* _xl_h = new float[N_elements];
    float* _xu_h = new float[N_elements];
    for (int i=0; i<N_elements; i++) {
        _xl_h[i] = xl;
        _xu_h[i] = xu;
    }
    cudaMemcpy(_xl,_xl_h, N_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_xu,_xu_h, N_elements * sizeof(float), cudaMemcpyHostToDevice);

    // initialize starting point
    float f_init = std::numeric_limits<float>::max();
    normMin_cuda(m,n,A,b,x,f_init,g, &assist_buffer_cuda, 'XS');

    // initialize number of bounds
    cudaMalloc(&_nbd, N_elements * sizeof(_nbd[0]));
    int* _nbd_h = new int[N_elements];
    for (int i=0; i<N_elements; i++) _nbd_h[i] = nbd;
    cudaMemcpy(_nbd,_nbd_h, N_elements * sizeof(int), cudaMemcpyHostToDevice);

    LBFGSB_CUDA_SUMMARY<float> summary;
    memset(&summary, 0, sizeof(summary));

    // start optimization
    auto start_time = std::chrono::steady_clock::now();
    lbfgsbcuda::lbfgsbminimize<float>(N_elements, state, lbfgsb_options, x, _nbd,
                                     _xl, _xu, summary);
    auto end_time = std::chrono::steady_clock::now();
    std::cout << std::endl << "Summary: " << std::endl;
    std::cout << "Total time: "
              << (std::chrono::duration<float, std::milli>(end_time - start_time)
                          .count() /
                  static_cast<float>(1))
              << " ms" << std::endl;
    std::cout << "Iteration time: "
              << (std::chrono::duration<float, std::milli>(end_time - start_time)
                          .count() /
                  static_cast<float>(summary.num_iteration))
              << " ms / iteration" << std::endl;
    std::cout << "residual f: " << static_cast<float>(summary.residual_f) << std::endl;
    std::cout << "residual g: " << static_cast<float>(summary.residual_g) << std::endl;
    std::cout << "residual x: " << static_cast<float>(summary.residual_x) << std::endl;
    std::cout << "iterations: " << static_cast<float>(summary.num_iteration) << std::endl;
    std::cout << "info: " << static_cast<float>(summary.info) << std::endl;

    // release allocated memory
    cudaFree(x);
    cudaFree(g);
    cudaFree(_xl);
    cudaFree(_xu);
    cudaFree(_nbd);
    cudaFree(assist_buffer_cuda);
    cudaFree(A);
    cudaFree(b);
    delete[] A_h;
    delete[] b_h;
    delete[] _xl_h;
    delete[] _xu_h;
    delete[] _nbd_h;

    // release cublas
    cublasDestroy(state.m_cublas_handle);
    return minimal_f;
}

int main(int argc, char* argv[]) {
  // here we have some problems
  std::cout << "problem size: 1000, constrained, CPU, single precision" << std::endl;
  float min_f_cpu_sgl = test_normMin_cpu<float>(1000, 1);
  std::cout << "CPU result " << min_f_cpu_sgl << std::endl << std::endl;

  std::cout << "problem size: 1000, constrained, GPU, single precision" << std::endl;
  float min_f_gpu_sgl = test_normMin_cuda(1000, 1);
  std::cout << "GPU result " << min_f_gpu_sgl << std::endl;

  // removing the bounds "solves" this problem
  std::cout << "problem size: 1000, unconstrained, CPU, single precision" << std::endl;
  min_f_cpu_sgl = test_normMin_cpu<float>(1000, 0);
  std::cout << "CPU result " << min_f_cpu_sgl << std::endl << std::endl;

  std::cout << "problem size: 1000, unconstrained, GPU, single precision" << std::endl;
  min_f_gpu_sgl = test_normMin_cuda(1000, 0);
  std::cout << "GPU result " << min_f_gpu_sgl << std::endl;

  return 0;
}
