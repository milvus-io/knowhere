
// -*- c++ -*-

#include "hook.h"

#include <iostream>
#include <mutex>

#include "distances_avx.h"
#include "distances_avx512.h"
#include "distances_ref.h"
#include "distances_sse.h"
#include "instruction_set.h"
#include "knowhere/log.h"
namespace faiss {

bool use_avx512 = true;
bool use_avx2 = true;
bool use_sse4_2 = true;

decltype(fvec_inner_product) fvec_inner_product = fvec_inner_product_ref;
decltype(fvec_L2sqr) fvec_L2sqr = fvec_L2sqr_ref;
decltype(fvec_L1) fvec_L1 = fvec_L1_ref;
decltype(fvec_Linf) fvec_Linf = fvec_Linf_ref;
decltype(fvec_norm_L2sqr) fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
decltype(fvec_L2sqr_ny) fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
decltype(fvec_inner_products_ny) fvec_inner_products_ny = fvec_inner_products_ny_ref;
decltype(fvec_madd) fvec_madd = fvec_madd_ref;
decltype(fvec_madd_and_argmin) fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

bool
cpu_support_avx512() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX512F() && instruction_set_inst.AVX512DQ() && instruction_set_inst.AVX512BW());
}

bool
cpu_support_avx2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX2());
}

bool
cpu_support_sse4_2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.SSE42());
}

void
fvec_hook(std::string& simd_type) {
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);
    simd_type = "REF";
    if (use_avx512 && cpu_support_avx512()) {
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
        return;
    }
    if (use_avx2 && cpu_support_avx2()) {
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
        return;
    }
    if (use_sse4_2 && cpu_support_sse4_2()) {
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
    }
}

static int init_hook_ = []() {
    std::string simd_type;
    fvec_hook(simd_type);
    KNOWHERE_INFO("simd type: {}", simd_type);
    return 0;
}();

}  // namespace faiss
