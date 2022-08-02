
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
#include <common/FileManager.h>
#include <index/vector_index/IndexAnnoy.h>
#include <index/vector_index/IndexBinaryIDMAP.h>
#include <index/vector_index/IndexHNSW.h>
#include <index/vector_index/IndexIDMAP.h>
#include <index/vector_index/IndexIVF.h>
#include <index/vector_index/IndexIVFSQ.h>
#include <index/vector_index/IndexIVFPQ.h>
#include <index/vector_index/IndexBinaryIVF.h>
#include <index/vector_index/IndexDiskANN.h>
#include <index/vector_index/IndexDiskANNConfig.h>
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

%ignore knowhere::IndexDiskANN::IndexDiskANN(std::string, MetricType, std::unique_ptr<FileManager>);

%include <std_string.i>
%include <std_pair.i>
%include <std_map.i>
%include <std_shared_ptr.i>

%shared_ptr(knowhere::Dataset)
%template(DatasetPtr) std::shared_ptr<Dataset>;
%shared_ptr(knowhere::IndexDiskANN<float>)
%shared_ptr(knowhere::IndexDiskANN<int8_t>)
%shared_ptr(knowhere::IndexDiskANN<uint8_t>)
%include <common/Dataset.h>
%include <utils/BitsetView.h>
%include <common/BinarySet.h>
%include <common/Config.h>
%include <common/Typedef.h>
%include <cache/DataObj.h>
%include <index/Index.h>
%include <index/VecIndex.h>
%include <index/vector_index/FaissBaseBinaryIndex.h>
%include <index/vector_offset_index/OffsetBaseIndex.h>
%include <index/vector_index/FaissBaseIndex.h>
%include <common/FileManager.h>
%include <index/vector_index/IndexAnnoy.h>
%include <index/vector_index/IndexHNSW.h>
%include <index/vector_index/IndexIVF.h>
%include <index/vector_index/IndexIVFSQ.h>
%include <index/vector_index/IndexIVFPQ.h>
%include <index/vector_index/IndexIDMAP.h>
%include <index/vector_index/IndexBinaryIDMAP.h>
%include <index/vector_index/IndexBinaryIVF.h>
%include <index/vector_index/IndexDiskANN.h>
%include <index/vector_index/IndexDiskANNConfig.h>
#ifdef KNOWHERE_GPU_VERSION
%include <index/vector_index/gpu/GPUIndex.h>
%include <index/vector_index/gpu/IndexGPUIVF.h>
%include <index/vector_index/gpu/IndexGPUIVFPQ.h>
%include <index/vector_index/gpu/IndexGPUIVFSQ.h>
%include <index/vector_index/gpu/IndexGPUIDMAP.h>
#endif
%include <index/vector_offset_index/IndexIVF_NM.h>


// Support for DiskANN
%template(IndexDiskANNf) knowhere::IndexDiskANN<float>;
%template(IndexDiskANNi8) knowhere::IndexDiskANN<int8_t>;
%template(IndexDiskANNui8) knowhere::IndexDiskANN<uint8_t>;

%inline %{

std::shared_ptr<knowhere::IndexDiskANN<float>>
buildDiskANNf(std::string index_prefix, std::string metric_type, std::shared_ptr<knowhere::FileManager> file_manager) {
    return std::make_shared<knowhere::IndexDiskANN<float>>(index_prefix, metric_type, 
            std::unique_ptr<FileManager>(file_manager.get()));
}

std::shared_ptr<knowhere::IndexDiskANN<int8_t>>
buildDiskANNi8(std::string index_prefix, std::string metric_type, std::shared_ptr<knowhere::FileManager> file_manager) {
    return std::make_shared<knowhere::IndexDiskANN<int8_t>>(index_prefix, metric_type, 
            std::unique_ptr<FileManager>(file_manager.get()));
}

std::shared_ptr<knowhere::IndexDiskANN<uint8_t>>
buildDiskANNui8(std::string index_prefix, std::string metric_type, std::shared_ptr<knowhere::FileManager> file_manager) {
    return std::make_shared<knowhere::IndexDiskANN<uint8_t>>(index_prefix, metric_type, 
            std::unique_ptr<FileManager>(file_manager.get()));
}

class FileManagerImpl: public knowhere::FileManager {
 public:
    bool
    LoadFile(const std::string& filename) noexcept { return true; }
    bool
    AddFile(const std::string& filename) noexcept { return true; }
    std::optional<bool>
    IsExisted(const std::string& filename) noexcept { return true; }
    bool
    RemoveFile(const std::string& filename) noexcept { return true; }
};

std::shared_ptr<knowhere::FileManager>
EmptyFileManager() {
    return std::make_shared<FileManagerImpl>();
}

%}

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
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int* xb, int nb, int dim)}
%apply (uint8_t *IN_ARRAY1, int DIM1) {(uint8_t *block, int size)}
%apply (int *IN_ARRAY1, int DIM1) {(int *lims, int len)}
%apply (int *IN_ARRAY1, int DIM1) {(int *ids, int len)}
%apply (float *IN_ARRAY1, int DIM1) {(float *dis, int len)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2){(float *dis,int nq_1,int k_1)}
%apply (int *INPLACE_ARRAY2, int DIM1, int DIM2){(int *ids,int nq_2,int k_2)}


%inline %{


knowhere::DatasetPtr
ArrayToDataSetFloat(float* xb, int nb, int dim) {
    auto ret_ds = std::make_shared<Dataset>();
    ret_ds->Set<int64_t>("rows", nb);
    ret_ds->Set<int64_t>("dim", dim);
    ret_ds->Set<const void*>("tensor", xb);
    return ret_ds;
};

knowhere::DatasetPtr
ArrayToDataSetInt(int *xb, int nb, int dim){
    auto ret_ds = std::make_shared<Dataset>();
    ret_ds->Set<int64_t>("rows", nb);
    ret_ds->Set<int64_t>("dim", dim * 32);
    ret_ds->Set<const void*>("tensor", xb);
    return ret_ds;
};

void
DumpResultDataSet(knowhere::Dataset& result, float* dis, int nq_1, int k_1, int* ids, int nq_2, int k_2) {
    auto ids_ = result.Get<const int64_t*>("ids");
    auto dist_ = result.Get<const float*>("distance");
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
DumpRangeResultIds(knowhere::Dataset& result, int* ids, int len) {
    auto ids_ = result.Get<const int64_t*>("ids");
    for (int i = 0; i < len; ++i) {
        *(ids + i) = *((int64_t*)(ids_) + i);
    }
}

void
DumpRangeResultLimits(knowhere::Dataset& result, int* lims, int len) {
    auto lims_ = result.Get<const size_t*>("lims");
    for (int i = 0; i < len; ++i) {
        *(lims + i) = *((int64_t*)(lims_) + i);
    }
}

void
DumpRangeResultDis(knowhere::Dataset& result, float* dis, int len) {
    auto dist_ = result.Get<const float*>("distance");
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

