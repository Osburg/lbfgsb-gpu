#ifndef EXAMPLES_NORMMIN_NORMMIN_CPU_H_
#define EXAMPLES_NORMMIN_NORMMIN_CPU_H_

template <typename real>
void normMin_cpu(int& m, int& n, real* A, real* b, real* x, real& f, real* fgrad, real** assist_buffer, int task);

#endif // EXAMPLES_NORMMIN_NORMMIN_CPU_H_
