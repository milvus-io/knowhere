
// -*- c++ -*-

#include "FaissHookFvec.h"

#include <iostream>
#include <mutex>
#if !defined(__APPLE__) || !defined(__aarch64__)
#include "cpuinfo_x86.h"
#endif

#if defined(__aarch64__)
#include "distances_simd.h"
#include "distances_simd_avx.h"
#include "distances_simd_avx512.h"
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

#if !defined(__APPLE__) || !defined(__aarch64__)
static const cpu_features::X86Features features = cpu_features::GetX86Info().features;
bool
cpu_support_avx512() {
    return (features.avx512f && features.avx512dq && features.avx512bw);
}

bool
cpu_support_avx2() {
    return (features.avx2);
}

bool
cpu_support_sse4_2() {
    return (features.sse4_2);
}
#else

bool cpu_support_avx512() {
    return false;
}

bool
cpu_support_avx2() {
    return false;
}

bool
cpu_support_sse4_2() {
    return false;
}

#endif

void
hook_fvec(std::string& simd_type) {
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);
    #if defined(__x86_64__)
    // fvec hook can be set outside
    if (faiss_use_avx512 && cpu_support_avx512()) {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_avx512;
        fvec_L2sqr = fvec_L2sqr_avx512;
        fvec_L1 = fvec_L1_avx512;
        fvec_Linf = fvec_Linf_avx512;

        simd_type = "AVX512";
    } else if (faiss_use_avx2 && cpu_support_avx2()) {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_avx;
        fvec_L2sqr = fvec_L2sqr_avx;
        fvec_L1 = fvec_L1_avx;
        fvec_Linf = fvec_Linf_avx;

        simd_type = "AVX2";
    } else if (faiss_use_sse4_2 && cpu_support_sse4_2()) {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_sse;
        fvec_L2sqr = fvec_L2sqr_sse;
        fvec_L1 = fvec_L1_sse;
        fvec_Linf = fvec_Linf_sse;

        simd_type = "SSE4_2";
    } else {
        /* for IVFFLAT */
        fvec_inner_product = fvec_inner_product_ref;
        fvec_L2sqr = fvec_L2sqr_ref;
        fvec_L1 = fvec_L1_ref;
        fvec_Linf = fvec_Linf_ref;

        simd_type = "REF";
    }
    #endif
    std::cout << "FAISS hook " << simd_type << std::endl;
}

}  // namespace faiss
