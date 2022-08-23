#include "ivf.h"

#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include "index/ivf/ivf_config.h"
#include "io/FaissIO.h"
#include "knowhere/knowhere.h"
namespace knowhere {
template <typename T>
Error
IVFIndexNode<T>::Build(const DataSet& dataset, const Config& cfg) {
    Train(dataset, cfg);
    Add(dataset, cfg);
    return Error::success;
}
template <typename T>
Error
IVFIndexNode<T>::Train(const DataSet& dataset, const Config& cfg) {
    const IVFConfig& ivf_cfg = static_cast<const IVFConfig&>(cfg);
    if (ivf_cfg.metric_type == "L2") {
        faiss::Index* qzr = new faiss::IndexFlat(ivf_cfg.dim, faiss::METRIC_L2);
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            this->index_ = new faiss::IndexIVFFlat(qzr, ivf_cfg.dim, ivf_cfg.nlist, faiss::METRIC_L2);
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            const IVFPQConfig& ivf_pq_cfg = static_cast<const IVFPQConfig&>(cfg);
            this->index_ = new faiss::IndexIVFPQ(qzr, ivf_cfg.dim, ivf_pq_cfg.nlist, ivf_pq_cfg.m, ivf_pq_cfg.nbits,
                                                 faiss::METRIC_L2);
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            this->index_ = new faiss::IndexIVFScalarQuantizer(qzr, ivf_cfg.dim, ivf_cfg.nlist,
                                                              faiss::QuantizerType::QT_8bit, faiss::METRIC_L2);
        }
    }

    if (ivf_cfg.metric_type == "IP") {
        faiss::Index* qzr = new faiss::IndexFlat(ivf_cfg.dim, faiss::METRIC_INNER_PRODUCT);
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            this->index_ = new faiss::IndexIVFFlat(qzr, ivf_cfg.dim, ivf_cfg.nlist, faiss::METRIC_INNER_PRODUCT);
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            const IVFPQConfig& ivf_pq_cfg = static_cast<const IVFPQConfig&>(cfg);
            this->index_ = new faiss::IndexIVFPQ(qzr, ivf_cfg.dim, ivf_pq_cfg.nlist, ivf_pq_cfg.m, ivf_pq_cfg.nbits,
                                                 faiss::METRIC_INNER_PRODUCT);
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            this->index_ = new faiss::IndexIVFScalarQuantizer(
                qzr, ivf_cfg.dim, ivf_cfg.nlist, faiss::QuantizerType::QT_8bit, faiss::METRIC_INNER_PRODUCT);
        }
    }

    if (!this->index_)
        return Error::empty_index;

    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    index_->train(rows, (const float*)data);
    return Error::success;
}

template <typename T>
Error
IVFIndexNode<T>::Add(const DataSet& dataset, const Config& cfg) {
    if (!this->index_)
        return Error::empty_index;
    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    index_->add(rows, (const float*)data);
    return Error::success;
}

template <typename T>
expected<DataSetPtr, Error>
IVFIndexNode<T>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_)
        return unexpected(Error::empty_index);
    if (!this->index_->is_trained)
        return unexpected(Error::index_not_trained);
    auto rows = dataset.GetRows();
    auto data = dataset.GetTensor();
    const IVFConfig& ivf_cfg = static_cast<const IVFConfig&>(cfg);
    int parallel_mode = -1;
    if (ivf_cfg.nprobe > 1 && rows <= 4) {
        parallel_mode = 1;
    } else {
        parallel_mode = 0;
    }
    size_t max_codes = 0;
    int64_t* ids(new (std::nothrow) int64_t[rows * ivf_cfg.k]);
    float* dis(new (std::nothrow) float[rows * ivf_cfg.k]);
    try {
        index_->search_thread_safe(rows, (const float*)data, ivf_cfg.k, dis, ids, ivf_cfg.nprobe, parallel_mode,
                                   max_codes, bitset);
    } catch (...) {
        std::unique_ptr<int64_t> ids_auto_delete(ids);
        std::unique_ptr<float> dis_auto_delete(dis);
        return unexpected(Error::faiss_inner_error);
    }
    auto results = std::make_shared<DataSet>();
    results->SetDim(ivf_cfg.dim);
    results->SetRows(rows);
    results->SetIds(ids);
    results->SetDistance(dis);
    return results;
}

template <typename T>
expected<DataSetPtr, Error>
IVFIndexNode<T>::SearchByRange(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_)
        return unexpected(Error::empty_index);
    if (!this->index_->is_trained)
        return unexpected(Error::index_not_trained);
    const IVFConfig& ivf_cfg = static_cast<const IVFConfig&>(cfg);
    auto rows = dataset.GetRows();
    auto data = dataset.GetTensor();
    faiss::RangeSearchResult res(rows);
    int parallel_mode = -1;
    if (ivf_cfg.nprobe > 1 && rows <= 4) {
        parallel_mode = 1;
    } else {
        parallel_mode = 0;
    }
    size_t max_codes = 0;

    try {
        index_->range_search_thread_safe(rows, (const float*)data, ivf_cfg.radius, &res, ivf_cfg.nprobe, parallel_mode,
                                         max_codes, bitset);
    } catch (...) {
        return unexpected(Error::faiss_inner_error);
    }
    auto result = std::make_shared<DataSet>();

    result->SetDim(ivf_cfg.dim);
    result->SetRows(rows);
    result->SetIds(res.labels);
    res.labels = nullptr;
    result->SetLims(res.lims);
    res.lims = nullptr;
    result->SetDistance(res.distances);
    res.distances = nullptr;
    return result;
}

template <typename T>
expected<DataSetPtr, Error>
IVFIndexNode<T>::GetVectorByIds(const DataSet& dataset, const Config& cfg) const {
    if (!this->index_)
        return unexpected(Error::empty_index);
    if (!this->index_->is_trained)
        return unexpected(Error::index_not_trained);
    auto rows = dataset.GetRows();
    const IVFConfig& ivf_cfg = static_cast<const IVFConfig&>(cfg);
    float* p_x(new (std::nothrow) float[ivf_cfg.dim * rows]);
    index_->make_direct_map(true);
    auto p_ids = dataset.GetIds();
    try {
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            assert(id >= 0 && id < index_->ntotal);
            index_->reconstruct(id, p_x + i * ivf_cfg.dim);
        }
    } catch (...) {
        std::unique_ptr<float> p_x_auto_delete(p_x);
        return unexpected(Error::faiss_inner_error);
    }

    auto results = std::make_shared<DataSet>();
    results->SetTensor(p_x);

    return results;
}

template <typename T>
Error
IVFIndexNode<T>::Serialization(BinarySet& binset) const {
    try {
        MemoryIOWriter writer;
        faiss::write_index(index_, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        binset.Append("IVF", data, writer.rp);
        return Error::success;
    } catch (...) {
        return Error::faiss_inner_error;
    }
}

template <typename T>
Error
IVFIndexNode<T>::Deserialization(const BinarySet& binset) {
    auto binary = binset.GetByName("IVF");

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();
    if (index_)
        delete index_;
    try {
        index_ = static_cast<T*>(faiss::read_index(&reader));
    } catch (...) {
        return Error::faiss_inner_error;
    }
    return Error::success;
}

KNOWHERE_REGISTER_GLOBAL(IVFFLAT, []() { return Index<IVFIndexNode<faiss::IndexIVFFlat>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(IVFPQ, []() { return Index<IVFIndexNode<faiss::IndexIVFPQ>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(IVFSQ, []() { return Index<IVFIndexNode<faiss::IndexIVFScalarQuantizer>>::Create(); });

}  // namespace knowhere
