#include <functional>
#include <map>

#include "common/metric.h"
#include "faiss/IndexFlat.h"
#include "faiss/gpu/GpuCloner.h"
#include "faiss/gpu/GpuIndexFlat.h"
#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/index_io.h"
#include "index/flat_gpu/flat_gpu_config.h"
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
    GpuFlatIndexNode(const Object& object) : gpu_index_(nullptr) {
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
    Train(const DataSet& dataset, const Config& cfg) override {
        return Status::success;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        const GpuFlatConfig& f_cfg = static_cast<const GpuFlatConfig&>(cfg);
        auto metric = Str2FaissMetricType(f_cfg.metric_type);
        if (!metric.has_value()) {
            LOG_KNOWHERE_WARNING_ << "metric type error, " << f_cfg.metric_type;
            return metric.error();
        }

        for (auto dev : f_cfg.gpu_ids) {
            this->devs_.push_back(dev);
            this->res_.push_back(new (std::nothrow) faiss::gpu::StandardGpuResources);
        }

        const void* x = dataset.GetTensor();
        const int64_t n = dataset.GetRows();
        faiss::Index* gpu_index = nullptr;
        try {
            auto host_index = std::make_unique<faiss::IndexFlat>(f_cfg.dim, metric.value());
            gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(this->res_, this->devs_, host_index.get());
            gpu_index->add(n, (const float*)x);
        } catch (const std::exception& e) {
            if (gpu_index)
                delete gpu_index;
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }
        if (this->gpu_index_)
            delete this->gpu_index_;
        this->gpu_index_ = gpu_index;
        return Status::success;
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!gpu_index_) {
            LOG_KNOWHERE_WARNING_ << "index not empty, deleted old index.";
            return unexpected(Status::empty_index);
        }

        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto nq = dataset.GetRows();
        auto x = dataset.GetTensor();
        auto len = f_cfg.k * nq;
        int64_t* ids = nullptr;
        float* dis = nullptr;
        try {
            ids = new (std::nothrow) int64_t[len];
            dis = new (std::nothrow) float[len];
            gpu_index_->search(nq, (const float*)x, f_cfg.k, dis, ids, bitset);
        } catch (const std::exception& e) {
            std::unique_ptr<int64_t[]> auto_delete_ids(ids);
            std::unique_ptr<float[]> auto_delete_dis(dis);
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return unexpected(Status::faiss_inner_error);
        }

        return GenResultDataSet(nq, f_cfg.k, ids, dis);
    }

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        DataSetPtr results = std::make_shared<DataSet>();
        auto nq = dataset.GetRows();
        auto in_ids = dataset.GetIds();
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        try {
            float* xq = new (std::nothrow) float[nq * f_cfg.dim];
            for (int64_t i = 0; i < nq; i++) {
                int64_t id = in_ids[i];
                gpu_index_->reconstruct(id, xq + i * f_cfg.dim);
            }
            results->SetTensor(xq);
            return results;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return unexpected(Status::faiss_inner_error);
        }
        return results;
    }

    virtual Status
    Save(BinarySet& binset) const override {
        if (!gpu_index_) {
            LOG_KNOWHERE_WARNING_ << "serilalization on empty index.";
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            std::unique_ptr<faiss::Index> host_index(faiss::gpu::index_gpu_to_cpu(gpu_index_));

            faiss::write_index(host_index.get(), &writer);
            std::shared_ptr<uint8_t[]> data(writer.data_);

            binset.Append("FLAT", data, writer.rp);

            size_t dev_s = this->devs_.size();
            uint8_t* buf = new uint8_t[sizeof(dev_s) + sizeof(int) * dev_s];
            auto device_id_ = std::shared_ptr<uint8_t[]>(buf);
            memcpy(buf, &dev_s, sizeof(dev_s));
            memcpy(buf + sizeof(dev_s), this->devs_.data(), sizeof(devs_[0]) * dev_s);
            binset.Append("device_ids", device_id_, sizeof(size_t) + sizeof(int) * dev_s);

        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    virtual Status
    Load(const BinarySet& binset, const Config& cfg) override {
        auto binary = binset.GetByName("FLAT");
        MemoryIOReader reader;
        try {
            reader.total = binary->size;
            reader.data_ = binary->data.get();
            std::unique_ptr<faiss::Index> index(faiss::read_index(&reader));

            size_t dev_s = 1;
            auto device_ids = binset.GetByName("device_ids");
            memcpy(&dev_s, device_ids->data.get(), sizeof(dev_s));
            this->devs_.resize(dev_s);
            memcpy(this->devs_.data(), device_ids->data.get() + sizeof(size_t), sizeof(int) * dev_s);
            for (size_t i = 0; i < dev_s; ++i)
                this->res_.push_back(new (std::nothrow) faiss::gpu::StandardGpuResources);
            gpu_index_ = faiss::gpu::index_cpu_to_gpu_multiple(this->res_, this->devs_, index.get());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<GpuFlatConfig>();
    }

    virtual int64_t
    Dim() const override {
        return gpu_index_->d;
    }

    virtual int64_t
    Size() const override {
        return gpu_index_->ntotal * gpu_index_->d * sizeof(float);
    }

    virtual int64_t
    Count() const override {
        return gpu_index_->ntotal;
    }

    virtual std::string
    Type() const override {
        return "GPUFLAT";
    }

    virtual ~GpuFlatIndexNode() {
        if (gpu_index_)
            delete gpu_index_;
    }

 private:
    std::vector<int> devs_;
    std::vector<faiss::gpu::GpuResourcesProvider*> res_;
    faiss::Index* gpu_index_;
};

KNOWHERE_REGISTER_GLOBAL(GPUFLAT, [](const Object& object) { return Index<GpuFlatIndexNode>::Create(object); });

}  // namespace knowhere
