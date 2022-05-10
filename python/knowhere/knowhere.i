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
#include <stdint.h>
#include <omp.h>
#include <memory>
#ifdef SWIGPYTHON
#undef popcount64
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif
#include <common/Dataset.h>
#include <index/vector_index/adapter/VectorAdapter.h>
#include <common/BinarySet.h>
#include <common/Utils.h>
#include <common/Config.h>
#include <common/Typedef.h>
#include <index/vector_index/IndexAnnoy.h>
#include <index/vector_index/IndexHNSW.h>
#include <index/vector_index/IndexIVF.h>
#include <index/vector_index/IndexIVFSQ.h>
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
%include <index/vector_index/adapter/VectorAdapter.h>
%include <utils/BitsetView.h>
%include <common/BinarySet.h>
%include <common/Utils.h>
%include <common/Config.h>
%include <common/Typedef.h>
%include <index/vector_index/IndexAnnoy.h>
%include <index/vector_index/IndexHNSW.h>
%include <index/vector_index/IndexIVF.h>
%include <index/vector_index/IndexIVFSQ.h>

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
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2){(float *dis,int nq_1,int k_1)}
%apply (int *INPLACE_ARRAY2, int DIM1, int DIM2){(int *ids,int nq_2,int k_2)}
%inline %{


knowhere::DatasetPtr ArrayToDataSet( float* xb,int nb, int dim){
    return knowhere::GenDataset(nb, dim, xb);
};

void DumpResultDataSet(knowhere::DatasetPtr result, float *dis, int nq_1, int k_1, 
                       int *ids,int nq_2, int k_2){
    auto ids_ = knowhere::GetDatasetIDs(result);
    auto dist_ = knowhere::GetDatasetDistance(result);
    assert(nq_1==nq_2);
    assert(k_1==k_2);
    for (int i = 0; i < nq_1; i++) {
        for (int j = 0; j < k_1; ++j) {
            *(ids+i*k_1+j) = *((int64_t*)(ids_) + i * k_1 + j);
            *(dis+i*k_1+j) = *((float*)(dist_) + i * k_1 + j);
        }
    }
}

knowhere::Config CreateConfig(std::string str){
    return knowhere::Config::parse(str);
}

faiss::BitsetView EmptyBitSetView(){
    return faiss::BitsetView(nullptr);
};

faiss::BitsetView ArrayToBitsetView(uint8_t *block, int size){
    return faiss::BitsetView(block, size);
}
%}

