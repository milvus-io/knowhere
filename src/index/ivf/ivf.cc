#include "common/metric.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/index_io.h"
#include "index/ivf/ivf_config.h"
#include "io/FaissIO.h"
#include "knowhere/knowhere.h"
namespace knowhere {

template <typename T>
struct QuantizerT {};

template <>
struct QuantizerT<faiss::IndexIVFFlat> {
    typedef faiss::IndexFlat* type;
};

template <>
struct QuantizerT<faiss::IndexIVFPQ> {
    typedef faiss::IndexFlat* type;
};
template <>
struct QuantizerT<faiss::IndexIVFScalarQuantizer> {
    typedef faiss::IndexFlat* type;
};
template <>
struct QuantizerT<faiss::IndexBinaryIVF> {
    typedef faiss::IndexBinaryFlat* type;
};

template <typename T>
class IvfIndexNode : public IndexNode {
 public:
    IvfIndexNode() : index_(nullptr), qzr_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexIVFFlat>::value || std::is_same<T, faiss::IndexIVFPQ>::value ||
                          std::is_same<T, faiss::IndexIVFScalarQuantizer>::value ||
                          std::is_same<T, faiss::IndexBinaryIVF>::value,
                      "not support.");
    }
    virtual Error
    Build(const DataSet& dataset, const Config& cfg) override;
    virtual Error
    Train(const DataSet& dataset, const Config& cfg) override;

    virtual Error
    Add(const DataSet& dataset, const Config& cfg) override;

    virtual expected<DataSetPtr, Error>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;
    virtual expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override;
    virtual Error
    Serialization(BinarySet& binset) const override;
    virtual Error
    Deserialization(const BinarySet& binset) override;
    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value)
            return std::make_unique<IvfFlatConfig>();
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value)
            return std::make_unique<IvfPqConfig>();
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value)
            return std::make_unique<IvfSqConfig>();
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value)
            return std::make_unique<IvfBinConfig>();
    };
    virtual int64_t
    Dims() const override {
        if (!index_)
            return -1;
        return index_->d;
    };
    virtual int64_t
    Size() const override {
        if (!index_)
            return 0;
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto nlist = index_->nlist;
            auto code_size = index_->code_size;
            return (nb * code_size + nb * sizeof(int64_t) + nlist * code_size);
        }

        if constexpr (std::is_same<T, faiss::IndexIVFPQ>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto code_size = index_->code_size;
            auto pq = index_->pq;
            auto nlist = index_->nlist;
            auto d = index_->d;

            auto capacity = nb * code_size + nb * sizeof(int64_t) + nlist * d * sizeof(float);
            auto centroid_table = pq.M * pq.ksub * pq.dsub * sizeof(float);
            auto precomputed_table = nlist * pq.M * pq.ksub * sizeof(float);
            return (capacity + centroid_table + precomputed_table);
        }
        if constexpr (std::is_same<T, faiss::IndexIVFScalarQuantizer>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto code_size = index_->code_size;
            auto nlist = index_->nlist;
            return (nb * code_size + nb * sizeof(int64_t) + 2 * code_size + nlist * code_size);
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto nlist = index_->nlist;
            auto code_size = index_->code_size;
            return (nb * code_size + nb * sizeof(int64_t) + nlist * code_size);
        }
    };
    virtual int64_t
    Count() const override {
        if (!index_)
            return 0;
        return index_->ntotal;
    };
    virtual std::string
    Type() const override {
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value)
            return "IVFFLAT";
        if constexpr (std::is_same<T, faiss::IndexIVFPQ>::value)
            return "IVFPQ";
        if constexpr (std::is_same<T, faiss::IndexIVFScalarQuantizer>::value)
            return "IVFSQ";
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value)
            return "IVFBIN";
    };
    virtual ~IvfIndexNode() {
        if (index_)
            delete index_;
        if (qzr_)
            delete qzr_;
    };

 private:
    T* index_;
    typename QuantizerT<T>::type qzr_;
};

}  // namespace knowhere

namespace knowhere {

template <typename T>
Error
IvfIndexNode<T>::Build(const DataSet& dataset, const Config& cfg) {
    auto err = Train(dataset, cfg);
    if (err != Error::success)
        return err;
    err = Add(dataset, cfg);
    return err;
}
template <typename T>
Error
IvfIndexNode<T>::Train(const DataSet& dataset, const Config& cfg) {
    const BaseConfig& base_cfg = static_cast<const IvfConfig&>(cfg);
    auto metric = Str2FaissMetricType(base_cfg.metric_type);
    if (!metric.has_value())
        return Error::invalid_metric_type;
    if (this->index_) {
        delete this->index_;
        this->index_ = nullptr;
    }
    if (this->qzr_) {
        delete this->qzr_;
        this->qzr_ = nullptr;
    }
    if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
        const IvfFlatConfig& ivf_flat_cfg = static_cast<const IvfFlatConfig&>(cfg);
        this->qzr_ = new (std::nothrow) faiss::IndexFlat(ivf_flat_cfg.dim, metric.value());
        this->index_ =
            new (std::nothrow) faiss::IndexIVFFlat(this->qzr_, ivf_flat_cfg.dim, ivf_flat_cfg.nlist, metric.value());
    }
    if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
        const IvfPqConfig& ivf_pq_cfg = static_cast<const IvfPqConfig&>(cfg);
        this->qzr_ = new (std::nothrow) faiss::IndexFlat(ivf_pq_cfg.dim, metric.value());
        this->index_ = new (std::nothrow) faiss::IndexIVFPQ(this->qzr_, ivf_pq_cfg.dim, ivf_pq_cfg.nlist, ivf_pq_cfg.m,
                                                            ivf_pq_cfg.nbits, metric.value());
    }
    if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
        const IvfSqConfig& ivf_sq_cfg = static_cast<const IvfSqConfig&>(cfg);
        this->qzr_ = new (std::nothrow) faiss::IndexFlat(ivf_sq_cfg.dim, metric.value());
        this->index_ = new (std::nothrow) faiss::IndexIVFScalarQuantizer(this->qzr_, ivf_sq_cfg.dim, ivf_sq_cfg.nlist,
                                                                         faiss::QuantizerType::QT_8bit, metric.value());
    }

    if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
        const IvfBinConfig& ivf_bin_cfg = static_cast<const IvfBinConfig&>(cfg);
        this->qzr_ = new faiss::IndexBinaryFlat(ivf_bin_cfg.dim, metric.value());
        this->index_ =
            new (std::nothrow) faiss::IndexBinaryIVF(this->qzr_, ivf_bin_cfg.dim, ivf_bin_cfg.nlist, metric.value());
    }

    if (!this->index_)
        return Error::empty_index;

    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
        index_->train(rows, (const uint8_t*)data);
    } else {
        index_->train(rows, (const float*)data);
    }
    return Error::success;
}

template <typename T>
Error
IvfIndexNode<T>::Add(const DataSet& dataset, const Config& cfg) {
    if (!this->index_)
        return Error::empty_index;
    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
        index_->add(rows, (const uint8_t*)data);
    } else {
        index_->add(rows, (const float*)data);
    }
    return Error::success;
}

template <typename T>
expected<DataSetPtr, Error>
IvfIndexNode<T>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_)
        return unexpected(Error::empty_index);
    if (!this->index_->is_trained)
        return unexpected(Error::index_not_trained);
    auto rows = dataset.GetRows();
    auto data = dataset.GetTensor();
    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);
    int parallel_mode = 0;
    if (ivf_cfg.nprobe > 1 && rows <= 4) {
        parallel_mode = 1;
    }
    int64_t* ids(new (std::nothrow) int64_t[rows * ivf_cfg.k]);
    float* dis(new (std::nothrow) float[rows * ivf_cfg.k]);
    try {
        size_t max_codes = 0;
        index_->search_thread_safe(rows, (const float*)data, ivf_cfg.k, dis, ids, ivf_cfg.nprobe, parallel_mode,
                                   max_codes, bitset);
    } catch (const std::exception&) {
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

template <>
expected<DataSetPtr, Error>
IvfIndexNode<faiss::IndexBinaryIVF>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_)
        return unexpected(Error::empty_index);
    if (!this->index_->is_trained)
        return unexpected(Error::index_not_trained);
    auto rows = dataset.GetRows();
    auto data = dataset.GetTensor();
    auto ivf_bin_cfg = static_cast<const IvfBinConfig&>(cfg);

    int64_t* ids(new (std::nothrow) int64_t[rows * ivf_bin_cfg.k]);
    float* dis(new (std::nothrow) float[rows * ivf_bin_cfg.k]);
    auto i_dis = reinterpret_cast<int32_t*>(dis);
    try {
        index_->search(rows, (const uint8_t*)data, ivf_bin_cfg.k, i_dis, ids, bitset);
    } catch (const std::exception&) {
        std::unique_ptr<int64_t> ids_auto_delete(ids);
        std::unique_ptr<float> dis_auto_delete(dis);
        return unexpected(Error::faiss_inner_error);
    }
    auto results = std::make_shared<DataSet>();
    results->SetDim(ivf_bin_cfg.dim);
    results->SetRows(rows);
    results->SetIds(ids);
    if (index_->metric_type == faiss::METRIC_Hamming) {
        int64_t num = rows * ivf_bin_cfg.k;
        for (int64_t i = 0; i < num; i++) {
            dis[i] = static_cast<float>(i_dis[i]);
        }
    }
    results->SetDistance(dis);
    return results;
}

template <typename T>
expected<DataSetPtr, Error>
IvfIndexNode<T>::GetVectorByIds(const DataSet& dataset, const Config& cfg) const {
    if (!this->index_)
        return unexpected(Error::empty_index);
    if (!this->index_->is_trained)
        return unexpected(Error::index_not_trained);
    auto rows = dataset.GetRows();
    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);
    float* p_x(new (std::nothrow) float[ivf_cfg.dim * rows]);
    index_->make_direct_map(true);
    auto p_ids = dataset.GetIds();
    try {
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            assert(id >= 0 && id < index_->ntotal);
            index_->reconstruct(id, p_x + i * ivf_cfg.dim);
        }
    } catch (const std::exception&) {
        std::unique_ptr<float> p_x_auto_delete(p_x);
        return unexpected(Error::faiss_inner_error);
    }

    auto results = std::make_shared<DataSet>();
    results->SetTensor(p_x);

    return results;
}

template <>
expected<DataSetPtr, Error>
IvfIndexNode<faiss::IndexBinaryIVF>::GetVectorByIds(const DataSet& dataset, const Config& cfg) const {
    if (!this->index_)
        return unexpected(Error::empty_index);
    if (!this->index_->is_trained)
        return unexpected(Error::index_not_trained);
    auto rows = dataset.GetRows();
    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);
    uint8_t* p_x(new (std::nothrow) uint8_t[ivf_cfg.dim * rows / 8]);
    index_->make_direct_map(true);
    auto p_ids = dataset.GetIds();
    try {
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            assert(id >= 0 && id < index_->ntotal);
            index_->reconstruct(id, p_x + i * ivf_cfg.dim / 8);
        }
    } catch (std::exception&) {
        std::unique_ptr<uint8_t> p_x_auto_delete(p_x);
        return unexpected(Error::faiss_inner_error);
    }

    auto results = std::make_shared<DataSet>();
    results->SetTensor(p_x);

    return results;
}

template <typename T>
Error
IvfIndexNode<T>::Serialization(BinarySet& binset) const {
    try {
        MemoryIOWriter writer;
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value)
            faiss::write_index_binary(index_, &writer);
        else
            faiss::write_index(index_, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            binset.Append("BinaryIVF", data, writer.rp);
        } else {
            binset.Append("IVF", data, writer.rp);
        }
        return Error::success;
    } catch (const std::exception&) {
        return Error::faiss_inner_error;
    }
}

template <typename T>
Error
IvfIndexNode<T>::Deserialization(const BinarySet& binset) {
    std::string name = "IVF";
    if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
        name = "BinaryIVF";
    }
    auto binary = binset.GetByName(name);

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();
    if (index_)
        delete index_;
    try {
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value)
            index_ = static_cast<T*>(faiss::read_index_binary(&reader));
        else
            index_ = static_cast<T*>(faiss::read_index(&reader));
    } catch (const std::exception&) {
        return Error::faiss_inner_error;
    }
    return Error::success;
}

KNOWHERE_REGISTER_GLOBAL(IVFBIN, []() { return Index<IvfIndexNode<faiss::IndexBinaryIVF>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(IVFFLAT, []() { return Index<IvfIndexNode<faiss::IndexIVFFlat>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(IVFPQ, []() { return Index<IvfIndexNode<faiss::IndexIVFPQ>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(IVFSQ, []() { return Index<IvfIndexNode<faiss::IndexIVFScalarQuantizer>>::Create(); });

}  // namespace knowhere
