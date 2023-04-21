
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

%ignore knowhere::IndexFactory;
%ignore knowhere::IndexNode;
%ignore knowhere::Index;
%ignore knowhere::expected;
%{
#include <stdint.h>
#include <memory>
#ifdef SWIGPYTHON
#undef popcount64
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif
#include <knowhere/expected.h>
#include <knowhere/factory.h>
#include <knowhere/comp/local_file_manager.h>
using namespace knowhere;
%}

%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%include "typemaps.i"
%init %{
import_array();
%}

%include <std_string.i>
%include <std_pair.i>
%include <std_map.i>
%include <std_shared_ptr.i>
%include <exception.i>
%shared_ptr(knowhere::DataSet)
%shared_ptr(knowhere::BinarySet)
%template(DataSetPtr) std::shared_ptr<knowhere::DataSet>;
%template(BinarySetPtr) std::shared_ptr<knowhere::BinarySet>;
%include <knowhere/expected.h>
%include <knowhere/dataset.h>
%include <knowhere/binaryset.h>
%include <knowhere/bitsetview.h>
%include <knowhere/expected.h>

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* xb, int nb, int dim)}
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int* xb, int nb, int dim)}
%apply (uint8_t *IN_ARRAY1, int DIM1) {(uint8_t *block, int size)}
%apply (int *IN_ARRAY1, int DIM1) {(int *lims, int len)}
%apply (int *IN_ARRAY1, int DIM1) {(int *ids, int len)}
%apply (float *IN_ARRAY1, int DIM1) {(float *dis, int len)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2){(float *dis,int nq_1,int k_1)}
%apply (int *INPLACE_ARRAY2, int DIM1, int DIM2){(int *ids,int nq_2,int k_2)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2){(float *data,int rows,int dim)}

%typemap(in, numinputs=0) knowhere::Status& status(knowhere::Status tmp) %{
    $1 = &tmp;
%}
%typemap(argout) knowhere::Status& status %{
    PyObject *o;
    o = PyInt_FromLong(long(*$1));
    $result = SWIG_Python_AppendOutput($result, o);
%}

%pythoncode %{
from enum import Enum
def redo(prefix):
    tmpD = { k : v for k,v in globals().items() if k.startswith(prefix + '_')}
    for k,v in tmpD.items():
        del globals()[k]
    tmpD = {k[len(prefix)+1:]:v for k,v in tmpD.items()}
    globals()[prefix] = Enum(prefix,tmpD)
redo('Status')
del redo
del Enum
%}

%inline %{
class IndexWrap {
 public:
    IndexWrap(const std::string& name) {
        if (name == std::string("DISKANN")) {
            std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
            auto diskann_pack = knowhere::Pack(file_manager);
            idx = IndexFactory::Instance().Create(name, diskann_pack);
        } else {
            idx = IndexFactory::Instance().Create(name);
        }
    }

    knowhere::Status
    Build(knowhere::DataSetPtr dataset, const std::string& json) {
        return idx.Build(*dataset, knowhere::Json::parse(json));
    }

    knowhere::Status
    Train(knowhere::DataSetPtr dataset, const std::string& json) {
        return idx.Train(*dataset, knowhere::Json::parse(json));
    }

    knowhere::Status
    Add(knowhere::DataSetPtr dataset, const std::string& json) {
        return idx.Add(*dataset, knowhere::Json::parse(json));
    }

    knowhere::DataSetPtr
    Search(knowhere::DataSetPtr dataset, const std::string& json, const knowhere::BitsetView& bitset, knowhere::Status& status) {
        auto res = idx.Search(*dataset, knowhere::Json::parse(json), bitset);
        if (res.has_value()) {
            status = knowhere::Status::success;
            return res.value();
        } else {
            status = res.error();
            return nullptr;
        }
    }

    knowhere::DataSetPtr
    RangeSearch(knowhere::DataSetPtr dataset, const std::string& json, const knowhere::BitsetView& bitset, knowhere::Status& status){
        auto res = idx.RangeSearch(*dataset, knowhere::Json::parse(json), bitset);
        if (res.has_value()) {
            status = knowhere::Status::success;
            return res.value();
        } else {
            status = res.error();
            return nullptr;
        }
    }

    knowhere::DataSetPtr
    GetVectorByIds(knowhere::DataSetPtr dataset, const std::string& json, knowhere::Status& status) {
        auto res = idx.GetVectorByIds(*dataset, knowhere::Json::parse(json));
        if (res.has_value()) {
            status = knowhere::Status::success;
            return res.value();
        } else {
            status = res.error();
            return nullptr;
        }
    }

    knowhere::Status
    Serialize(knowhere::BinarySetPtr binset) {
        return idx.Serialize(*binset);
    }

    knowhere::Status
    Deserialize(knowhere::BinarySetPtr binset) {
        return idx.Deserialize(*binset);
    }

    int64_t
    Dim() {
        return idx.Dim();
    }

    int64_t
    Size() {
        return idx.Size();
    }

    int64_t
    Count() {
        return idx.Count();
    }

    std::string
    Type() {
        return idx.Type();
    }

 private:
    Index<IndexNode> idx;
};

class BitSet {
 public:
    BitSet(const int num_bits) : num_bits_(num_bits) {
        bit_set_.resize(num_bits_ / 8, 0);
    }

    void
    SetBit(const int idx) {
        bit_set_[idx >> 3] |= 0x1 << (idx & 0x7);
    }

    knowhere::BitsetView
    GetBitSetView() {
        return knowhere::BitsetView(bit_set_.data(), num_bits_);
    }

 private:
    std::vector<uint8_t> bit_set_;
    int num_bits_ = 0;
};

knowhere::BitsetView
GetNullBitSetView() {
    return nullptr;
};

knowhere::DataSetPtr
Array2DataSetF(float* xb, int nb, int dim) {
    auto ds = std::make_shared<DataSet>();
    ds->SetIsOwner(false);
    ds->SetRows(nb);
    ds->SetDim(dim);
    ds->SetTensor(xb);
    return ds;
};

knowhere::DataSetPtr
Array2DataSetI(int *xb, int nb, int dim){
    auto ds = std::make_shared<DataSet>();
    ds->SetIsOwner(false);
    ds->SetRows(nb);
    ds->SetDim(dim*32);
    ds->SetTensor(xb);
    return ds;
};

knowhere::DataSetPtr
Array2DataSetIds(int* ids, int len){
    auto ds = std::make_shared<DataSet>();
    ds->SetIsOwner(true);
    ds->SetRows(len);

    int64_t* ids_ = new int64_t[len];
    for (int i = 0; i < len; i++) {
        ids_[i] = (int64_t)ids[i];
    }
    ds->SetIds(ids_);
    return ds;
};

int64_t DataSet_Rows(knowhere::DataSetPtr results){
    return results->GetRows();
}

int64_t DataSet_Dim(knowhere::DataSetPtr results){
    return results->GetDim();
}

knowhere::BinarySetPtr GetBinarySet() {
    return std::make_shared<knowhere::BinarySet>();
}

knowhere::DataSetPtr GetNullDataSet() {
    return nullptr;
}

void
DataSet2Array(knowhere::DataSetPtr result, float* dis, int nq_1, int k_1, int* ids, int nq_2, int k_2) {
    auto ids_ = result->GetIds();
    auto dist_ = result->GetDistance();
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
DataSetTensor2Array(knowhere::DataSetPtr result, float* data, int rows, int dim) {
    auto data_ = result->GetTensor();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < dim; ++j) {
            *(data + i * dim + j) = *((float*)(data_) + i * dim + j);
        }
    }
}

void
DumpRangeResultIds(knowhere::DataSetPtr result, int* ids, int len) {
    auto ids_ = result->GetIds();
    for (int i = 0; i < len; ++i) {
        *(ids + i) = *((int64_t*)(ids_) + i);
    }
}

void
DumpRangeResultLimits(knowhere::DataSetPtr result, int* lims, int len) {
    auto lims_ = result->GetLims();
    for (int i = 0; i < len; ++i) {
        *(lims + i) = *((size_t*)(lims_) + i);
    }
}

void
DumpRangeResultDis(knowhere::DataSetPtr result, float* dis, int len) {
    auto dist_ = result->GetDistance();
    for (int i = 0; i < len; ++i) {
        *(dis + i) = *((float*)(dist_) + i);
    }
}

%}
