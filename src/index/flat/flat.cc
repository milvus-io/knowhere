#include <functional>
#include <map>

#include "common/metric.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"
#include "index/flat/flat_config.h"
#include "io/FaissIO.h"
#include "knowhere/knowhere.h"
namespace knowhere {

template <typename T>
class FlatIndexNode : public IndexNode {
 public:
    FlatIndexNode() : index_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexFlat>::value || std::is_same<T, faiss::IndexBinaryFlat>::value,
                      "not suppprt.");
    }
    virtual Error
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Error::success)
            return err;
        err = Add(dataset, cfg);
        return err;
    }
    virtual Error
    Train(const DataSet& dataset, const Config& cfg) override {
        return Error::success;
    }
    virtual Error
    Add(const DataSet& dataset, const Config& cfg) override {
        if (!index_) {
            delete index_;
            index_ = nullptr;
        }
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto metric = Str2FaissMetricType(f_cfg.metric_type);
        if (!metric.has_value())
            return metric.error();
        index_ = new (std::nothrow) T(f_cfg.dim, metric.value());

        const void* x = dataset.GetTensor();
        const int64_t n = dataset.GetRows();
        if constexpr (std::is_same<T, faiss::IndexFlat>::value)
            index_->add(n, (const float*)x);
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value)
            index_->add(n, (const uint8_t*)x);
        return Error::success;
    }
    virtual expected<DataSetPtr, Error>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            return unexpected(Error::empty_index);
        }

        DataSetPtr results = std::make_shared<DataSet>();
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto nq = dataset.GetRows();
        auto x = dataset.GetTensor();
        auto len = f_cfg.k * nq;
        int64_t* ids = nullptr;
        float* dis = nullptr;
        try {
            ids = new (std::nothrow) int64_t[len];
            dis = new (std::nothrow) float[len];
            if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
                index_->search(nq, (const float*)x, f_cfg.k, dis, ids, bitset);
            }
            if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
                auto i_dis = reinterpret_cast<int32_t*>(dis);
                index_->search(nq, (const uint8_t*)x, f_cfg.k, i_dis, ids, bitset);
                if (index_->metric_type == faiss::METRIC_Hamming) {
                    int64_t num = nq * f_cfg.k;
                    for (int64_t i = 0; i < num; i++) {
                        dis[i] = static_cast<float>(i_dis[i]);
                    }
                }
            }
        } catch (const std::exception&) {
            std::unique_ptr<int64_t[]> auto_delete_ids(ids);
            std::unique_ptr<float[]> auto_delete_dis(dis);
            return unexpected(Error::faiss_inner_error);
        }

        results->SetIds(ids);
        results->SetDistance(dis);
        return results;
    }

    virtual expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        DataSetPtr results = std::make_shared<DataSet>();
        auto nq = dataset.GetRows();
        auto in_ids = dataset.GetIds();
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            try {
                float* xq = new (std::nothrow) float[nq * f_cfg.dim];
                for (int64_t i = 0; i < nq; i++) {
                    int64_t id = in_ids[i];
                    index_->reconstruct(id, xq + i * f_cfg.dim);
                }
                results->SetTensor(xq);
                return results;
            } catch (const std::exception&) {
                return unexpected(Error::faiss_inner_error);
            }
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            try {
                uint8_t* xq = new (std::nothrow) uint8_t[nq * f_cfg.dim / 8];
                for (int64_t i = 0; i < nq; i++) {
                    int64_t id = in_ids[i];
                    index_->reconstruct(id, xq + i * f_cfg.dim / 8);
                }
                results->SetTensor(xq);
                return results;
            } catch (const std::exception&) {
                return unexpected(Error::faiss_inner_error);
            }
        }
    }
    virtual Error
    Serialization(BinarySet& binset) const override {
        if (!index_)
            return Error::empty_index;
        try {
            MemoryIOWriter writer;
            if constexpr (std::is_same<T, faiss::IndexFlat>::value)
                faiss::write_index(index_, &writer);
            if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value)
                faiss::write_index_binary(index_, &writer);
            std::shared_ptr<uint8_t[]> data(writer.data_);
            if constexpr (std::is_same<T, faiss::IndexFlat>::value)
                binset.Append("FLAT", data, writer.rp);
            if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value)
                binset.Append("BinaryIVF", data, writer.rp);
            return Error::success;
        } catch (const std::exception&) {
            return Error::faiss_inner_error;
        }
    }
    virtual Error
    Deserialization(const BinarySet& binset) override {
        if (index_) {
            delete index_;
            index_ = nullptr;
        }
        std::string name = "";
        if constexpr (std::is_same<T, faiss::IndexFlat>::value)
            name = "FLAT";
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value)
            name = "BinaryIVF";
        auto binary = binset.GetByName(name);

        MemoryIOReader reader;
        reader.total = binary->size;
        reader.data_ = binary->data.get();
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            faiss::Index* index = faiss::read_index(&reader);
            index_ = static_cast<T*>(index);
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            faiss::IndexBinary* index = faiss::read_index_binary(&reader);
            index_ = static_cast<T*>(index);
        }
        return Error::success;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<FlatConfig>();
    }
    virtual int64_t
    Dims() const override {
        return index_->d;
    }
    virtual int64_t
    Size() const override {
        return index_->ntotal * index_->d * sizeof(float);
    }
    virtual int64_t
    Count() const override {
        return index_->ntotal;
    }
    virtual std::string
    Type() const override {
        if constexpr (std::is_same<T, faiss::IndexFlat>::value)
            return "FLAT";
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value)
            return "BINFLAT";
    }
    virtual ~FlatIndexNode() {
        if (index_)
            delete index_;
    }

 private:
    T* index_;
};

KNOWHERE_REGISTER_GLOBAL(FLAT, []() { return Index<FlatIndexNode<faiss::IndexFlat>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(BINFLAT, []() { return Index<FlatIndexNode<faiss::IndexBinaryFlat>>::Create(); });

}  // namespace knowhere
   //
