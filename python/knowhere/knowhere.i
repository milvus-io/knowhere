
%module swigknowhere;

#pragma SWIG nowarn=321
#pragma SWIG nowarn=403
#pragma SWIG nowarn=325
#pragma SWIG nowarn=389
#pragma SWIG nowarn=341
#pragma SWIG nowarn=512
#pragma SWIG nowarn=362

%include <stdint.i>
typedef uint64_t size_t;
#define __restrict


%{
#include <omp.h>
#include <stdint.h>

#include <memory>
#ifdef SWIGPYTHON
#undef popcount64
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif
#include <common/BinarySet.h>
#include <common/Config.h>
#include <common/Dataset.h>
#include <common/Typedef.h>
#include <index/vector_index/IndexAnnoy.h>
#include <index/vector_index/IndexBinaryIDMAP.h>
#include <index/vector_index/IndexHNSW.h>
#include <index/vector_index/IndexIDMAP.h>
#include <index/vector_index/IndexIVF.h>
#include <index/vector_index/IndexIVFSQ.h>
#include <index/vector_index/IndexIVFPQ.h>
#include <index/vector_index/IndexBinaryIVF.h>
#ifdef KNOWHERE_GPU_VERSION
#include <index/vector_index/gpu/IndexGPUIVF.h>
#include <index/vector_index/gpu/IndexGPUIVFPQ.h>
#include <index/vector_index/gpu/IndexGPUIVFSQ.h>
#include <index/vector_index/gpu/IndexGPUIDMAP.h>
#endif
#include <index/vector_offset_index/IndexIVF_NM.h>
#include <archive/KnowhereConfig.h>
using namespace knowhere;
%}

%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%include <std_string.i>
%include <std_pair.i>
%include <std_map.i>
%include <std_shared_ptr.i>
%include <common/Dataset.h>
%include <utils/BitsetView.h>
%include <common/BinarySet.h>
%include <common/Config.h>
%include <common/Typedef.h>
%include <index/vector_index/IndexAnnoy.h>
%include <index/vector_index/IndexHNSW.h>
%include <index/vector_index/IndexIVF.h>
%include <index/vector_index/IndexIVFSQ.h>
%include <index/vector_index/IndexIVFPQ.h>
%include <index/vector_index/IndexIDMAP.h>
%include <index/vector_index/IndexBinaryIDMAP.h>
%include <index/vector_index/IndexBinaryIVF.h>
#ifdef KNOWHERE_GPU_VERSION
%include <index/vector_index/gpu/IndexGPUIVF.h>
%include <index/vector_index/gpu/IndexGPUIVFPQ.h>
%include <index/vector_index/gpu/IndexGPUIVFSQ.h>
%include <index/vector_index/gpu/IndexGPUIDMAP.h>
#endif
%include <index/vector_offset_index/IndexIVF_NM.h>

%shared_ptr(knowhere::Dataset)

#ifdef SWIGPYTHON
%define DOWNCAST(subclass)
    if (dynamic_cast<knowhere::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ ## subclass,$owner);
    } else
%enddef
#endif

%typemap(out) knowhere::VecIndex * {
DOWNCAST ( IndexAnnoy )
}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* xb, int nb, int dim)}
%apply (uint8_t *IN_ARRAY1, int DIM1) {(uint8_t *block, int size)}
%apply (int *IN_ARRAY1, int DIM1) {(int *lims, int len)}
%apply (int *IN_ARRAY1, int DIM1) {(int *ids, int len)}
%apply (float *IN_ARRAY1, int DIM1) {(float *dis, int len)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2){(float *dis,int nq_1,int k_1)}
%apply (int *INPLACE_ARRAY2, int DIM1, int DIM2){(int *ids,int nq_2,int k_2)}


%inline %{


knowhere::DatasetPtr
ArrayToDataSet(float* xb, int nb, int dim) {
    auto ret_ds = std::make_shared<Dataset>();
    ret_ds->Set<int64_t>("rows", nb);
    ret_ds->Set<int64_t>("dim", dim);
    ret_ds->Set<const void*>("tensor", xb);
    return ret_ds;
};

void
DumpResultDataSet(const knowhere::DatasetPtr& result, float* dis, int nq_1, int k_1, int* ids, int nq_2, int k_2) {
    auto ids_ = result->Get<const int64_t*>("ids");
    auto dist_ = result->Get<const float*>("distance");
    assert(nq_1 == nq_2);
    assert(k_1 == k_2);
    for (int i = 0; i < nq_1; i++) {
        for (int j = 0; j < k_1; ++j) {
            *(ids + i * k_1 + j) = *((int64_t*)(ids_) + i * k_1 + j);
            *(dis + i * k_1 + j) = *((float*)(dist_) + i * k_1 + j);
        }
    }
}

void
DumpRangeResultIds(const knowhere::DatasetPtr& result, int* ids, int len) {
    auto ids_ = result->Get<const int64_t*>("ids");
    for (int i = 0; i < len; ++i) {
        *(ids + i) = *((int64_t*)(ids_) + i);
    }
}

void
DumpRangeResultLimits(const knowhere::DatasetPtr& result, int* lims, int len) {
    auto lims_ = result->Get<const size_t*>("lims");
    for (int i = 0; i < len; ++i) {
        *(lims + i) = *((int64_t*)(lims_) + i);
    }
}

void
DumpRangeResultDis(const knowhere::DatasetPtr& result, float* dis, int len) {
    auto dist_ = result->Get<const float*>("distance");
    for (int i = 0; i < len; ++i) {
        *(dis + i) = *((float*)(dist_) + i);
    }
}

knowhere::Config
CreateConfig(const std::string& str) {
    auto cfg = knowhere::Config::parse(str);
    auto metric_type = cfg.at("metric_type").get<std::string>();
    std::transform(metric_type.begin(), metric_type.end(), metric_type.begin(), toupper);
    cfg["metric_type"] = metric_type;
    return cfg;
}

void
SetSimdType(const std::string& str) {
    if (str == "auto") {
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);
    }
    if (str == "avx512") {
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX512);
    }
    if (str == "avx2") {
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);
    }
    if (str == "sse4_2" || str == "avx") {
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::SSE4_2);
    }
}

faiss::BitsetView
EmptyBitSetView() {
    return faiss::BitsetView(nullptr);
};

faiss::BitsetView
ArrayToBitsetView(uint8_t* block, int size) {
    return faiss::BitsetView(block, size);
}

#ifdef KNOWHERE_GPU_VERSION
void
InitGpuResource(int dev_id, int pin_mem, int temp_mem, int res_num) {
    knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(dev_id, pin_mem, temp_mem, res_num);
}

void
ReleaseGpuResource() {
    knowhere::FaissGpuResourceMgr::GetInstance().Free();
}
#endif

%}

