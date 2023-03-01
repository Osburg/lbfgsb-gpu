#ifndef EXAMPLES_NORMAPPROXIMATION_NORMAPPROXIMATION_CUDA_H_
#define EXAMPLES_NORMAPPROXIMATION_NORMAPPROXIMATION_CUDA_H_

void normMin_cuda(int& m, int& n, float* A, float* b, float* x, float& f, float* fgrad, float** assist_buffer, int task);

#endif // EXAMPLES_NORMAPPROXIMATION_NORMAPPROXIMATION_CUDA_H_
