
// -*- c++ -*-

#include <iostream>
#include <mutex>

#include "FaissHookFvec.h"
#include "distances_simd.h"
#include "distances_simd_avx.h"
#include "distances_simd_avx512.h"
#include "distances_simd_sse.h"
#ifdef __linux__
#include "instruction_set.h"
#endif

namespace faiss {

bool faiss_use_avx512 = true;
bool faiss_use_avx2 = true;
bool faiss_use_sse4_2 = true;

/* set default to AVX */
fvec_func_ptr fvec_inner_product = fvec_inner_product_ref;
fvec_func_ptr fvec_L2sqr = fvec_L2sqr_ref;
fvec_func_ptr fvec_L1 = fvec_L1_ref;
fvec_func_ptr fvec_Linf = fvec_Linf_ref;
fvec_norm_L2sqr_func_ptr fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
fvec_L2sqr_ny_func_ptr fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
fvec_inner_products_ny_func_ptr fvec_inner_products_ny = fvec_inner_products_ny_ref;
fvec_madd_func_ptr fvec_madd = fvec_madd_ref;
fvec_madd_and_argmin_func_ptr fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

/*****************************************************************************/
#ifdef __linux__
bool cpu_support_avx512() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX512F() &&
            instruction_set_inst.AVX512DQ() &&
            instruction_set_inst.AVX512BW());
}

bool cpu_support_avx2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX2());
}

bool cpu_support_sse4_2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.SSE42());
}
#endif

void hook_fvec(std::string& simd_type) {
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);

#ifdef __linux__
    // fvec hook can be set outside
    if (faiss_use_avx512 && cpu_support_avx512()) {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_avx512;
        fvec_L2sqr = fvec_L2sqr_avx512;
        fvec_L1 = fvec_L1_avx512;
        fvec_Linf = fvec_Linf_avx512;

        fvec_norm_L2sqr = fvec_norm_L2sqr_sse;
        fvec_L2sqr_ny = fvec_L2sqr_ny_sse;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_sse;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        simd_type = "AVX512";
    } else if (faiss_use_avx2 && cpu_support_avx2()) {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_avx;
        fvec_L2sqr = fvec_L2sqr_avx;
        fvec_L1 = fvec_L1_avx;
        fvec_Linf = fvec_Linf_avx;

        fvec_norm_L2sqr = fvec_norm_L2sqr_sse;
        fvec_L2sqr_ny = fvec_L2sqr_ny_sse;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_sse;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        simd_type = "AVX2";
    } else if (faiss_use_sse4_2 && cpu_support_sse4_2()) {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_sse;
        fvec_L2sqr = fvec_L2sqr_sse;
        fvec_L1 = fvec_L1_sse;
        fvec_Linf = fvec_Linf_sse;

        fvec_norm_L2sqr = fvec_norm_L2sqr_sse;
        fvec_L2sqr_ny = fvec_L2sqr_ny_sse;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_sse;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        simd_type = "SSE4_2";
    } else {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_ref;
        fvec_L2sqr = fvec_L2sqr_ref;
        fvec_L1 = fvec_L1_ref;
        fvec_Linf = fvec_Linf_ref;

        fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
        fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
        fvec_inner_products_ny = fvec_inner_products_ny_ref;
        fvec_madd = fvec_madd_ref;
        fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

        simd_type = "REF";
    }
#else
    simd_type = "REF";
#endif
}

} // namespace faiss
