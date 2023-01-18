#ifndef HOOK_H
#define HOOK_H

#include <string>
namespace faiss {

extern float (*fvec_inner_product)(const float*, const float*, size_t);
extern float (*fvec_L2sqr)(const float*, const float*, size_t);
extern float (*fvec_L1)(const float*, const float*, size_t);
extern float (*fvec_Linf)(const float*, const float*, size_t);
extern float (*fvec_norm_L2sqr)(const float*, size_t);
extern void (*fvec_L2sqr_ny)(float*, const float*, const float*, size_t, size_t);
extern void (*fvec_inner_products_ny)(float*, const float*, const float*, size_t, size_t);
extern void (*fvec_madd)(size_t, const float*, float, const float*, float*);
extern int (*fvec_madd_and_argmin)(size_t, const float*, float, const float*, float*);

#if defined(__x86_64__)
extern bool use_avx512;
extern bool use_avx2;
extern bool use_sse4_2;
#endif

#if defined(__x86_64__)
bool
cpu_support_avx512();
bool
cpu_support_avx2();
bool
cpu_support_sse4_2();
#endif

void
fvec_hook(std::string&);

}  // namespace faiss

#endif /* HOOK_H */
