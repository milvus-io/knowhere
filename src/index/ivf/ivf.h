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
class IVFIndexNode : public IndexNode {
 public:
    IVFIndexNode() : index_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexIVFFlat>::value || std::is_same<T, faiss::IndexIVFPQ>::value ||
                          std::is_same<T, faiss::IndexIVFScalarQuantizer>::value,
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
    SearchByRange(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;
    virtual expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override;
    virtual Error
    Serialization(BinarySet& binset) const override;
    virtual Error
    Deserialization(const BinarySet& binset) override;
    virtual std::unique_ptr<Config>
    CreateConfig() const override {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value)
            return std::make_unique<IVFFLATConfig>();
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value)
            return std::make_unique<IVFPQConfig>();
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value)
            return std::make_unique<IVFSQConfig>();
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
        return 0;
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
        return "IVF";
    };
    virtual ~IVFIndexNode() {
        if (index_)
            delete index_;
    };

 private:
    T* index_;
};

}  // namespace knowhere
