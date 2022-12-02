#include "common/metric.h"
#include "common/range_util.h"
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
    FlatIndexNode(const Object& object) : index_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexFlat>::value || std::is_same<T, faiss::IndexBinaryFlat>::value,
                      "not suppprt.");
    }

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success)
            return err;
        err = Add(dataset, cfg);
        return err;
    }

    virtual Status
    Train(const DataSet&, const Config&) override {
        return Status::success;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        T* index = nullptr;
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto metric = Str2FaissMetricType(f_cfg.metric_type);
        if (!metric.has_value()) {
            LOG_KNOWHERE_WARNING_ << "please check metric type, " << f_cfg.metric_type;
            return metric.error();
        }
        index = new (std::nothrow) T(dataset.GetDim(), metric.value());

        if (index == nullptr) {
            LOG_KNOWHERE_WARNING_ << "memory malloc error";
            return Status::malloc_error;
        }

        if (this->index_) {
            delete this->index_;
            LOG_KNOWHERE_WARNING_ << "index not empty, deleted old index";
        }
        this->index_ = index;
        const void* x = dataset.GetTensor();
        const int64_t n = dataset.GetRows();
        if constexpr (std::is_same<T, faiss::IndexFlat>::value)
            index_->add(n, (const float*)x);
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value)
            index_->add(n, (const uint8_t*)x);
        return Status::success;
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return unexpected(Status::empty_index);
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
        } catch (const std::exception& e) {
            std::unique_ptr<int64_t[]> auto_delete_ids(ids);
            std::unique_ptr<float[]> auto_delete_dis(dis);
            LOG_KNOWHERE_WARNING_ << "error inner faiss, " << e.what();
            return unexpected(Status::faiss_inner_error);
        }

        return GenResultDataSet(nq, f_cfg.k, ids, dis);
    }

    virtual expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "range search on empty index.";
            return unexpected(Status::empty_index);
        }

        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto nq = dataset.GetRows();
        auto xq = dataset.GetTensor();

        int64_t* ids = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;
        try {
            float low_bound = f_cfg.radius_low_bound;
            float high_bound = f_cfg.radius_high_bound;

            faiss::RangeSearchResult res(nq);
            if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
                bool is_ip = (index_->metric_type == faiss::METRIC_INNER_PRODUCT);
                float radius = (is_ip ? low_bound : high_bound);
                index_->range_search(nq, (const float*)xq, radius, &res, bitset);
                GetRangeSearchResult(res, is_ip, nq, low_bound, high_bound, distances, ids, lims, bitset);
            }
            if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
                index_->range_search(nq, (const uint8_t*)xq, high_bound, &res, bitset);
                GetRangeSearchResult(res, false, nq, low_bound, high_bound, distances, ids, lims, bitset);
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss, " << e.what();
            return unexpected(Status::faiss_inner_error);
        }

        return GenResultDataSet(nq, ids, distances, lims);
    }

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        DataSetPtr results = std::make_shared<DataSet>();
        auto nq = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto in_ids = dataset.GetIds();
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            try {
                float* xq = new (std::nothrow) float[nq * dim];
                for (int64_t i = 0; i < nq; i++) {
                    int64_t id = in_ids[i];
                    index_->reconstruct(id, xq + i * dim);
                }
                results->SetTensor(xq);
                return results;
            } catch (const std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "error inner faiss, " << e.what();
                return unexpected(Status::faiss_inner_error);
            }
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            try {
                uint8_t* xq = new (std::nothrow) uint8_t[nq * dim / 8];
                for (int64_t i = 0; i < nq; i++) {
                    int64_t id = in_ids[i];
                    index_->reconstruct(id, xq + i * dim / 8);
                }
                results->SetTensor(xq);
                return results;
            } catch (const std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "error inner faiss, " << e.what();
                return unexpected(Status::faiss_inner_error);
            }
        }
    }

    virtual expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }

    virtual Status
    Serialize(BinarySet& binset) const override {
        if (!index_)
            return Status::empty_index;
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
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss, " << e.what();
            return Status::faiss_inner_error;
        }
    }

    virtual Status
    Deserialize(const BinarySet& binset) override {
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
        return Status::success;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<FlatConfig>();
    }

    virtual int64_t
    Dim() const override {
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

KNOWHERE_REGISTER_GLOBAL(FLAT,
                         [](const Object& object) { return Index<FlatIndexNode<faiss::IndexFlat>>::Create(object); });
KNOWHERE_REGISTER_GLOBAL(BINFLAT, [](const Object& object) {
    return Index<FlatIndexNode<faiss::IndexBinaryFlat>>::Create(object);
});
KNOWHERE_REGISTER_GLOBAL(BIN_FLAT, [](const Object& object) {
    return Index<FlatIndexNode<faiss::IndexBinaryFlat>>::Create(object);
});

}  // namespace knowhere
   //
