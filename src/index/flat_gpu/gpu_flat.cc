#include <functional>
#include <map>

#include "common/metric.h"
#include "faiss/IndexFlat.h"
#include "faiss/gpu/GpuCloner.h"
#include "faiss/gpu/GpuIndexFlat.h"
#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/index_io.h"
#include "index/flat/flat_config.h"
#include "io/FaissIO.h"
#include "knowhere/knowhere.h"
namespace knowhere {

static faiss::gpu::StandardGpuResources*
GetGpuRes() {
    static faiss::gpu::StandardGpuResources res;
    return &res;
}

class GpuFlatIndexNode : public IndexNode {
 public:
    GpuFlatIndexNode() : index_(nullptr) {
    }
    virtual Error
    Build(const DataSet& dataset, const Config& cfg) override {
        return Error::success;
    }
    virtual Error
    Train(const DataSet& dataset, const Config& cfg) override {
        return Error::success;
    }
    virtual Error
    Add(const DataSet& dataset, const Config& cfg) override {
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto metric = Str2FaissMetricType(f_cfg.metric_type);
        if (!metric.has_value())
            return metric.error();
        index_ = new (std::nothrow)
            faiss::gpu::GpuIndexFlat(GetGpuRes(), new (std::nothrow) faiss::IndexFlat(f_cfg.dim, metric.value()));

        const void* x = dataset.GetTensor();
        const int64_t n = dataset.GetRows();
        try {
            index_->add(n, (const float*)x);
        } catch (const std::exception& e) {
            return Error::faiss_inner_error;
        }
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
            index_->search(nq, (const float*)x, f_cfg.k, dis, ids, bitset);
        } catch (const std::exception& e) {
            std::unique_ptr<int64_t[]> auto_delete_ids(ids);
            std::unique_ptr<float[]> auto_delete_dis(dis);
            return unexpected(Error::faiss_inner_error);
        }

        results->SetIds(ids);
        results->SetDistance(dis);
        return results;
    }
    virtual expected<DataSetPtr, Error>
    SearchByRange(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return unexpected(Error::not_implemented);
    }

    virtual expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        DataSetPtr results = std::make_shared<DataSet>();
        auto nq = dataset.GetRows();
        auto in_ids = dataset.GetIds();
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        try {
            float* xq = new (std::nothrow) float[nq * f_cfg.dim];
            for (int64_t i = 0; i < nq; i++) {
                int64_t id = in_ids[i];
                index_->reconstruct(id, xq + i * f_cfg.dim);
            }
            results->SetTensor(xq);
            return results;
        } catch (...) {
            return unexpected(Error::faiss_inner_error);
        }
        return results;
    }
    virtual Error
    Serialization(BinarySet& binset) const override {
        if (!index_)
            return Error::empty_index;
        try {
            MemoryIOWriter writer;
            std::unique_ptr<faiss::Index> host_index(faiss::gpu::index_gpu_to_cpu(index_));

            faiss::write_index(host_index.get(), &writer);
            std::shared_ptr<uint8_t[]> data(writer.data_);

            binset.Append("FLAT", data, writer.rp);

        } catch (const std::exception& e) {
            return Error::faiss_inner_error;
        }
        return Error::success;
    }
    virtual Error
    Deserialization(const BinarySet& binset) override {
        auto binary = binset.GetByName("FLAT");
        MemoryIOReader reader;
        try {
            reader.total = binary->size;
            reader.data_ = binary->data.get();
            std::unique_ptr<faiss::Index> index(faiss::read_index(&reader));
            this->index_ =
                static_cast<faiss::gpu::GpuIndexFlat*>(faiss::gpu::index_cpu_to_gpu(GetGpuRes(), 0, index.get()));
        } catch (const std::exception& e) {
            return Error::faiss_inner_error;
        }

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
        return "GPUFLAT";
    }
    virtual ~GpuFlatIndexNode() {
        if (index_)
            delete index_;
    }

 private:
    faiss::gpu::GpuIndexFlat* index_;
};

KNOWHERE_REGISTER_GLOBAL(GPUFLAT, []() { return Index<GpuFlatIndexNode>::Create(); });

}  // namespace knowhere
