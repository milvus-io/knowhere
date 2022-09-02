#include "common/metric.h"
#include "faiss/gpu/GpuCloner.h"
#include "faiss/gpu/GpuIndexIVFFlat.h"
#include "faiss/gpu/GpuIndexIVFPQ.h"
#include "faiss/gpu/GpuIndexIVFScalarQuantizer.h"
#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/index_io.h"
#include "index/ivf_gpu/ivf_gpu_config.h"
#include "io/FaissIO.h"
#include "knowhere/knowhere.h"
namespace knowhere {

static faiss::gpu::StandardGpuResources*
GetGpuRes() {
    static faiss::gpu::StandardGpuResources res;
    return &res;
}

template <typename T>
struct FaissConfigType {};

template <>
struct FaissConfigType<faiss::gpu::GpuIndexIVFFlat> {
    typedef faiss::gpu::GpuIndexIVFFlatConfig Type;
};
template <>
struct FaissConfigType<faiss::gpu::GpuIndexIVFPQ> {
    typedef faiss::gpu::GpuIndexIVFPQConfig Type;
};
template <>
struct FaissConfigType<faiss::gpu::GpuIndexIVFScalarQuantizer> {
    typedef faiss::gpu::GpuIndexIVFScalarQuantizerConfig Type;
};

template <typename T>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<faiss::gpu::GpuIndexIVFFlat> {
    typedef IvfGpuFlatConfig Type;
};
template <>
struct KnowhereConfigType<faiss::gpu::GpuIndexIVFPQ> {
    typedef IvfGpuPqConfig Type;
};
template <>
struct KnowhereConfigType<faiss::gpu::GpuIndexIVFScalarQuantizer> {
    typedef IvfGpuSqConfig Type;
};

template <typename T>
class IvfGpuIndexNode : public IndexNode {
 public:
    IvfGpuIndexNode() : index_(nullptr) {
        static_assert(std::is_same<T, faiss::gpu::GpuIndexIVFFlat>::value ||
                      std::is_same<T, faiss::gpu::GpuIndexIVFPQ>::value ||
                      std::is_same<T, faiss::gpu::GpuIndexIVFScalarQuantizer>::value);
    }
    virtual Error
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Error::success)
            return err;
        return Add(dataset, cfg);
    }
    virtual Error
    Train(const DataSet& dataset, const Config& cfg) override {
        if (index_)
            delete index_;

        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        auto ivf_gpu_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);

        typename FaissConfigType<T>::Type faiss_cfg;
        faiss_cfg.device = ivf_gpu_cfg.gpu_id;
        auto metric = Str2FaissMetricType(ivf_gpu_cfg.metric_type);
        if (!metric.has_value())
            return metric.error();
        try {
            if constexpr (std::is_same<T, faiss::gpu::GpuIndexIVFFlat>::value)
                index_ =
                    new (std::nothrow) T(GetGpuRes(), ivf_gpu_cfg.dim, ivf_gpu_cfg.nlist, metric.value(), faiss_cfg);
            if constexpr (std::is_same<T, faiss::gpu::GpuIndexIVFPQ>::value)
                index_ = new (std::nothrow) T(GetGpuRes(), ivf_gpu_cfg.dim, ivf_gpu_cfg.nlist, ivf_gpu_cfg.m,
                                              ivf_gpu_cfg.nbits, metric.value(), faiss_cfg);
            if constexpr (std::is_same<T, faiss::gpu::GpuIndexIVFScalarQuantizer>::value)
                index_ = new (std::nothrow) T(GetGpuRes(), ivf_gpu_cfg.dim, ivf_gpu_cfg.nlist,
                                              faiss::QuantizerType::QT_8bit, metric.value(), true, faiss_cfg);
            index_->train(rows, reinterpret_cast<const float*>(tensor));
        } catch (...) {
            return Error::faiss_inner_error;
        }
        return Error::success;
    }
    virtual Error
    Add(const DataSet& dataset, const Config& cfg) override {
        if (!index_)
            return Error::empty_index;
        if (!index_->is_trained)
            return Error::index_not_trained;
        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        try {
            index_->add(rows, (const float*)tensor);
        } catch (...) {
            return Error::faiss_inner_error;
        }
        return Error::success;
    }
    virtual expected<DataSetPtr, Error>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto ivf_gpu_cfg = static_cast<const IvfGpuFlatConfig&>(cfg);
        index_->nprobe = ivf_gpu_cfg.nprobe;
        std::cout << index_->nprobe << std::endl;
        constexpr int64_t block_size = 2048;
        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        float* dis = new (std::nothrow) float[rows * ivf_gpu_cfg.k];
        int64_t* ids = new (std::nothrow) int64_t[rows * ivf_gpu_cfg.k];
        try {
            for (int i = 0; i < rows; i += block_size) {
                int64_t search_size = (rows - i > block_size) ? block_size : (rows - i);
                index_->search(search_size, reinterpret_cast<const float*>(tensor) + i * ivf_gpu_cfg.dim, ivf_gpu_cfg.k,
                               dis + i * ivf_gpu_cfg.k, ids + i * ivf_gpu_cfg.k, bitset);
            }
        } catch (std::exception& e) {
            std::unique_ptr<float> auto_delete_dis(dis);
            std::unique_ptr<int64_t> auto_delete_ids(ids);
            std::cout << e.what() << std::endl;
            return unexpected(Error::faiss_inner_error);
        }
        auto results = std::make_shared<DataSet>();
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
        return unexpected(Error::not_implemented);
    }
    virtual Error
    Serialization(BinarySet& binset) const override {
        if (!index_)
            return Error::empty_index;
        if (!index_->is_trained)
            return Error::index_not_trained;

        try {
            MemoryIOWriter writer;
            {
                faiss::Index* host_index = faiss::gpu::index_gpu_to_cpu(index_);
                faiss::write_index(host_index, &writer);
                delete host_index;
            }
            std::shared_ptr<uint8_t[]> data(writer.data_);

            binset.Append("IVF", data, writer.rp);

        } catch (...) {
            return Error::faiss_inner_error;
        }

        return Error::success;
    }
    virtual Error
    Deserialization(const BinarySet& binset) override {
        auto binary = binset.GetByName("IVF");
        MemoryIOReader reader;
        try {
            reader.total = binary->size;
            reader.data_ = binary->data.get();

            std::unique_ptr<faiss::Index> index(faiss::read_index(&reader));
            if (index_)
                delete index_;
            index_ = static_cast<T*>(faiss::gpu::index_cpu_to_gpu(GetGpuRes(), 0, index.get()));
        } catch (...) {
            return Error::faiss_inner_error;
        }
        return Error::success;
    }

    virtual std::unique_ptr<Config>
    CreateConfig() const override {
        return std::make_unique<IvfGpuFlatConfig>();
    }
    virtual int64_t
    Dims() const override {
        if (index_)
            return index_->d;
        return 0;
    }
    virtual int64_t
    Size() const override {
        auto nl = index_->getNumLists();
        size_t nb = 0;
        for (int i = 0; i < nl; ++i) nb += index_->getListIndices(i).size();
        auto nlist = index_->nlist;
        constexpr int code_size = sizeof(float);
        return (nb * code_size + nb * sizeof(int64_t) + nlist * code_size);
    }
    virtual int64_t
    Count() const override {
        if (index_)
            return index_->ntotal;
        return 0;
    }
    virtual std::string
    Type() const override {
        if constexpr (std::is_same<faiss::gpu::GpuIndexIVFFlat, T>::value) {
            return "GPUIVFFLAT";
        }
        if constexpr (std::is_same<faiss::gpu::GpuIndexIVFPQ, T>::value) {
            return "GPUIVFPQ";
        }
        if constexpr (std::is_same<faiss::gpu::GpuIndexIVFScalarQuantizer, T>::value) {
            return "GPUIVFSQ";
        }
    }
    virtual ~IvfGpuIndexNode() {
        if (index_)
            delete index_;
    }

 private:
    T* index_;
};
KNOWHERE_REGISTER_GLOBAL(GPUIVFFLAT, []() { return Index<IvfGpuIndexNode<faiss::gpu::GpuIndexIVFFlat>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(GPUIVFPQ, []() { return Index<IvfGpuIndexNode<faiss::gpu::GpuIndexIVFPQ>>::Create(); });
KNOWHERE_REGISTER_GLOBAL(GPUIVFSQ,
                         []() { return Index<IvfGpuIndexNode<faiss::gpu::GpuIndexIVFScalarQuantizer>>::Create(); });
}  // namespace knowhere
