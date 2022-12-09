#include "common/metric.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexReplicas.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/gpu/GpuCloner.h"
#include "faiss/gpu/GpuIndexIVF.h"
#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/index_io.h"
#include "index/ivf_gpu/ivf_gpu_config.h"
#include "io/FaissIO.h"
#include "knowhere/factory.h"

namespace knowhere {

template <typename T>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<faiss::IndexIVFFlat> {
    typedef IvfGpuFlatConfig Type;
};
template <>
struct KnowhereConfigType<faiss::IndexIVFPQ> {
    typedef IvfGpuPqConfig Type;
};
template <>
struct KnowhereConfigType<faiss::IndexIVFScalarQuantizer> {
    typedef IvfGpuSqConfig Type;
};

template <typename T>
class IvfGpuIndexNode : public IndexNode {
 public:
    IvfGpuIndexNode(const Object& object) : devs_({}), res_{}, gpu_index_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexIVFFlat>::value || std::is_same<T, faiss::IndexIVFPQ>::value ||
                      std::is_same<T, faiss::IndexIVFScalarQuantizer>::value);
    }

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success)
            return err;
        return Add(dataset, cfg);
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) override {
        if (gpu_index_ && gpu_index_->is_trained) {
            LOG_KNOWHERE_WARNING_ << "index is already trained";
            return Status::index_already_trained;
        }

        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        auto ivf_gpu_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);

        for (size_t i = 0; i < ivf_gpu_cfg.gpu_ids.size(); ++i) {
            this->devs_.push_back(i);
            this->res_.push_back(new (std::nothrow) faiss::gpu::StandardGpuResources);
        }

        auto metric = Str2FaissMetricType(ivf_gpu_cfg.metric_type);
        if (!metric.has_value()) {
            KNOWHERE_WARN("please check metric value, {}", ivf_gpu_cfg.metric_type);
            return metric.error();
        }
        faiss::Index* gpu_index;
        try {
            auto qzr = new (std::nothrow) faiss::IndexFlat(ivf_gpu_cfg.dim, metric.value());
            if (qzr == nullptr) {
                LOG_KNOWHERE_WARNING_ << "memory malloc error";
                return Status::malloc_error;
            }
            std::unique_ptr<faiss::IndexFlat> auto_delele_qzr(qzr);
            T* host_index = nullptr;
            if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
                host_index =
                    new (std::nothrow) faiss::IndexIVFFlat(qzr, ivf_gpu_cfg.dim, ivf_gpu_cfg.nlist, metric.value());
                if (host_index == nullptr) {
                    LOG_KNOWHERE_WARNING_ << "memory malloc error";
                    return Status::malloc_error;
                }
            }
            if constexpr (std::is_same<T, faiss::IndexIVFPQ>::value) {
                host_index = new (std::nothrow) faiss::IndexIVFPQ(qzr, ivf_gpu_cfg.dim, ivf_gpu_cfg.nlist,
                                                                  ivf_gpu_cfg.m, ivf_gpu_cfg.nbits, metric.value());
                if (host_index == nullptr) {
                    LOG_KNOWHERE_WARNING_ << "memory malloc error";
                    return Status::malloc_error;
                }
            }
            if constexpr (std::is_same<T, faiss::IndexIVFScalarQuantizer>::value) {
                host_index = new (std::nothrow) faiss::IndexIVFScalarQuantizer(
                    qzr, ivf_gpu_cfg.dim, ivf_gpu_cfg.nlist, faiss::QuantizerType::QT_8bit, metric.value());
                if (host_index == nullptr) {
                    LOG_KNOWHERE_WARNING_ << "memory malloc error";
                    return Status::malloc_error;
                }
            }
            std::unique_ptr<T> auto_delete_host_index(host_index);
            gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(this->res_, this->devs_, host_index);
            gpu_index->train(rows, reinterpret_cast<const float*>(tensor));

        } catch (std::exception& e) {
            if (gpu_index)
                delete gpu_index;
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }
        this->gpu_index_ = gpu_index;
        return Status::success;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        if (!gpu_index_)
            return Status::empty_index;
        if (!gpu_index_->is_trained)
            return Status::index_not_trained;
        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        try {
            gpu_index_->add(rows, (const float*)tensor);
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto ivf_gpu_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);
        if (auto ix = dynamic_cast<faiss::IndexReplicas*>(gpu_index_)) {
            for (int i = 0; i < ix->count(); ++i) {
                auto idx = dynamic_cast<faiss::gpu::GpuIndexIVF*>(ix->at(i));
                assert(idx != nullptr);
                idx->setNumProbes(ivf_gpu_cfg.nprobe);
            }
        }
        if (auto ix = dynamic_cast<faiss::gpu::GpuIndexIVF*>(gpu_index_)) {
            ix->setNumProbes(ivf_gpu_cfg.nprobe);
        }
        constexpr int64_t block_size = 2048;
        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        float* dis = new (std::nothrow) float[rows * ivf_gpu_cfg.k];
        int64_t* ids = new (std::nothrow) int64_t[rows * ivf_gpu_cfg.k];
        try {
            for (int i = 0; i < rows; i += block_size) {
                int64_t search_size = (rows - i > block_size) ? block_size : (rows - i);
                gpu_index_->search(search_size, reinterpret_cast<const float*>(tensor) + i * ivf_gpu_cfg.dim,
                                   ivf_gpu_cfg.k, dis + i * ivf_gpu_cfg.k, ids + i * ivf_gpu_cfg.k, bitset);
            }
        } catch (std::exception& e) {
            std::unique_ptr<float> auto_delete_dis(dis);
            std::unique_ptr<int64_t> auto_delete_ids(ids);
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return unexpected(Status::faiss_inner_error);
        }

        return GenResultDataSet(rows, ivf_gpu_cfg.k, ids, dis);
    }

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }

    virtual Status
    Serialize(BinarySet& binset) const override {
        if (!this->gpu_index_)
            return Status::empty_index;
        if (!this->gpu_index_->is_trained)
            return Status::index_not_trained;

        try {
            MemoryIOWriter writer;
            {
                faiss::Index* host_index = faiss::gpu::index_gpu_to_cpu(this->gpu_index_);
                faiss::write_index(host_index, &writer);
                delete host_index;
            }
            std::shared_ptr<uint8_t[]> data(writer.data_);

            binset.Append("IVF", data, writer.rp);
            size_t dev_s = this->devs_.size();
            uint8_t* buf = new uint8_t[sizeof(dev_s) + sizeof(int) * dev_s];
            auto device_id_ = std::shared_ptr<uint8_t[]>(buf);
            memcpy(buf, &dev_s, sizeof(dev_s));
            memcpy(buf + sizeof(dev_s), this->devs_.data(), sizeof(devs_[0]) * dev_s);
            binset.Append("device_ids", device_id_, sizeof(size_t) + sizeof(int) * dev_s);
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    virtual Status
    Deserialize(const BinarySet& binset) override {
        auto binary = binset.GetByName("IVF");
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
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<typename KnowhereConfigType<T>::Type>();
    }

    virtual int64_t
    Dim() const override {
        if (gpu_index_)
            return gpu_index_->d;
        return 0;
    }

    virtual int64_t
    Size() const override {
        return 0;
    }

    virtual int64_t
    Count() const override {
        if (gpu_index_)
            return gpu_index_->ntotal;
        return 0;
    }

    virtual std::string
    Type() const override {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            return "GPUIVFFLAT";
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            return "GPUIVFPQ";
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            return "GPUIVFSQ";
        }
    }

    virtual ~IvfGpuIndexNode() {
        if (gpu_index_)
            delete gpu_index_;
        for (auto&& p : res_) delete p;
    }

 private:
    std::vector<int> devs_;
    std::vector<faiss::gpu::GpuResourcesProvider*> res_;
    faiss::Index* gpu_index_;
};

KNOWHERE_REGISTER_GLOBAL(GPUIVFFLAT, [](const Object& object) {
    return Index<IvfGpuIndexNode<faiss::IndexIVFFlat>>::Create(object);
});
KNOWHERE_REGISTER_GLOBAL(GPUIVFPQ, [](const Object& object) {
    return Index<IvfGpuIndexNode<faiss::IndexIVFPQ>>::Create(object);
});
KNOWHERE_REGISTER_GLOBAL(GPUIVFSQ, [](const Object& object) {
    return Index<IvfGpuIndexNode<faiss::IndexIVFScalarQuantizer>>::Create(object);
});
}  // namespace knowhere
