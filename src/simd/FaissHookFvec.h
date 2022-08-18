
// -*- c++ -*-

#pragma once

#include <string>

namespace faiss {

typedef float (*fvec_func_ptr)(const float*, const float*, size_t);
typedef float (*fvec_norm_L2sqr_func_ptr)(const float*, size_t);
typedef void (*fvec_L2sqr_ny_func_ptr)(float*, const float*, const float*, size_t, size_t);
typedef void (*fvec_inner_products_ny_func_ptr)(float*, const float*, const float*, size_t, size_t);
typedef void (*fvec_madd_func_ptr)(size_t, const float*, float, const float*, float*);
typedef int (*fvec_madd_and_argmin_func_ptr)(size_t, const float*, float, const float*, float*);

extern bool faiss_use_avx512;
extern bool faiss_use_avx2;
extern bool faiss_use_sse4_2;

extern fvec_func_ptr fvec_inner_product;
extern fvec_func_ptr fvec_L2sqr;
extern fvec_func_ptr fvec_L1;
extern fvec_func_ptr fvec_Linf;
extern fvec_norm_L2sqr_func_ptr fvec_norm_L2sqr;
extern fvec_L2sqr_ny_func_ptr fvec_L2sqr_ny;
extern fvec_inner_products_ny_func_ptr fvec_inner_products_ny;
extern fvec_madd_func_ptr fvec_madd;
extern fvec_madd_and_argmin_func_ptr fvec_madd_and_argmin;

#ifdef __linux__
bool cpu_support_avx512();
bool cpu_support_avx2();
bool cpu_support_sse4_2();
#endif

void hook_fvec(std::string&);

} // namespace faiss
