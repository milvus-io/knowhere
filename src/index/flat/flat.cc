#include <functional>
#include <map>

#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"
#include "index/flat/flat_config.h"
#include "knowhere/knowhere.h"
#include "src/io/FaissIO.h"
namespace knowhere {

class FlatIndexNode : public IndexNode {
 public:
    FlatIndexNode() : index_(nullptr) {
    }
    virtual Error
    Build(const DataSet& dataset, const Config& cfg) override {
        Train(dataset, cfg);
        Add(dataset, cfg);
        return Error::success;
    }
    virtual Error
    Train(const DataSet& dataset, const Config& cfg) override {
        return Error::success;
    }
    virtual Error
    Add(const DataSet& dataset, const Config& cfg) override {
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        if (f_cfg.metric_type == "L2")
            index_ = new (std::nothrow) faiss::IndexFlat(f_cfg.dim, faiss::METRIC_L2);
        if (f_cfg.metric_type == "IP")
            index_ = new (std::nothrow) faiss::IndexFlat(f_cfg.dim, faiss::METRIC_INNER_PRODUCT);

        if (!index_) {
            const void* x = dataset.GetTensor();
            const int64_t n = dataset.GetRows();
            index_->add(n, (const float*)x);
            return Error::success;
        }
        return Error::invalid_metric_type;
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
        int64_t* ids = new (std::nothrow) int64_t[len];
        float* dis = new (std::nothrow) float[len];
        index_->search(nq, (const float*)x, f_cfg.k, dis, ids, bitset);

        results->SetIds(ids);
        results->SetDistance(dis);
        return results;
    }
    virtual expected<DataSetPtr, Error>
    SearchByRange(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            return unexpected(Error::empty_index);
        }
        DataSetPtr results = std::make_shared<DataSet>();
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto nq = dataset.GetRows();
        auto x = dataset.GetTensor();
        faiss::RangeSearchResult res(nq);
        index_->range_search(nq, (const float*)x, f_cfg.radius, &res, nullptr);
        results->SetIds(res.labels);
        results->SetDistance(res.distances);
        results->SetLims(res.lims);
        res.labels = nullptr;
        res.distances = nullptr;
        res.lims = nullptr;
        return results;
    }

    virtual expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        DataSetPtr results = std::make_shared<DataSet>();
        auto nq = dataset.GetRows();
        auto in_ids = dataset.GetIds();
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        float* xq = new (std::nothrow) float[nq * f_cfg.dim];
        for (int64_t i = 0; i < nq; i++) {
            int64_t id = in_ids[i];
            index_->reconstruct(id, xq + i * f_cfg.dim);
        }
        results->SetTensor(xq);
        return results;
    }
    virtual Error
    Serialization(BinarySet& binset) const override {
        if (!index_)
            return Error::empty_index;
        MemoryIOWriter writer;

        faiss::write_index(index_, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        binset.Append("FLAT", data, writer.rp);
        return Error::success;
    }
    virtual Error
    Deserialization(const BinarySet& binset) override {
        if (!index_)
            return Error::empty_index;
        auto binary = binset.GetByName("FLAT");

        MemoryIOReader reader;
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        faiss::Index* index = faiss::read_index(&reader);
        index_ = static_cast<faiss::IndexFlat*>(index);
        return Error::success;
    }

    virtual std::unique_ptr<Config>
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
        return "FLAT";
    }
    virtual ~FlatIndexNode() {
        if (index_)
            delete index_;
    }

 private:
    faiss::IndexFlat* index_;
};

KNOWHERE_REGISTER_GLOBAL(FLAT, []() { return Index<FlatIndexNode>::Create(); });

}  // namespace knowhere
   //
