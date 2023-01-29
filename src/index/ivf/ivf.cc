// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "common/metric.h"
#include "common/range_util.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/index_io.h"
#include "index/ivf/ivf_config.h"
#include "io/FaissIO.h"
#include "knowhere/factory.h"
#include "knowhere/feder/IVFFlat.h"
#include "knowhere/index_node_thread_pool_wrapper.h"

namespace knowhere {

#define RAW_DATA "RAW_DATA"

template <typename T>
struct QuantizerT {
    typedef faiss::IndexFlat type;
};

template <>
struct QuantizerT<faiss::IndexBinaryIVF> {
    using type = faiss::IndexBinaryFlat;
};

template <typename T>
class IvfIndexNode : public IndexNode {
 public:
    IvfIndexNode(const Object& object) : index_(nullptr), qzr_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexIVFFlat>::value || std::is_same<T, faiss::IndexIVFPQ>::value ||
                          std::is_same<T, faiss::IndexIVFScalarQuantizer>::value ||
                          std::is_same<T, faiss::IndexBinaryIVF>::value,
                      "not support.");
    }
    Status
    Build(const DataSet& dataset, const Config& cfg) override;
    Status
    Train(const DataSet& dataset, const Config& cfg) override;
    Status
    Add(const DataSet& dataset, const Config& cfg) override;
    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;
    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;
    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override;
    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }
    Status
    Serialize(BinarySet& binset) const override;
    Status
    Deserialize(const BinarySet& binset) override;
    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            return std::make_unique<IvfFlatConfig>();
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            return std::make_unique<IvfPqConfig>();
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            return std::make_unique<IvfSqConfig>();
        }
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            return std::make_unique<IvfBinConfig>();
        }
    };
    int64_t
    Dim() const override {
        if (!index_) {
            return -1;
        }
        return index_->d;
    };
    int64_t
    Size() const override {
        if (!index_) {
            return 0;
        }
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
    int64_t
    Count() const override {
        if (!index_) {
            return 0;
        }
        return index_->ntotal;
    };
    std::string
    Type() const override {
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
        }
        if constexpr (std::is_same<T, faiss::IndexIVFPQ>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFPQ;
        }
        if constexpr (std::is_same<T, faiss::IndexIVFScalarQuantizer>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT;
        }
    };
    ~IvfIndexNode() override {
        if (index_) {
            delete index_;
        }
        if (qzr_) {
            delete qzr_;
        }
    };

 private:
    T* index_;
    typename QuantizerT<T>::type* qzr_;
};

}  // namespace knowhere

namespace knowhere {

template <typename T>
Status
IvfIndexNode<T>::Build(const DataSet& dataset, const Config& cfg) {
    auto err = Train(dataset, cfg);
    if (err != Status::success) {
        return err;
    }
    err = Add(dataset, cfg);
    return err;
}

inline int64_t
MatchNlist(int64_t size, int64_t nlist) {
    const int64_t MIN_POINTS_PER_CENTROID = 39;

    if (nlist * MIN_POINTS_PER_CENTROID > size) {
        // nlist is too large, adjust to a proper value
        LOG_KNOWHERE_WARNING_ << "nlist(" << nlist << ") is too large, adjust to a proper value";
        nlist = std::max(static_cast<int64_t>(1), size / MIN_POINTS_PER_CENTROID);
        LOG_KNOWHERE_WARNING_ << "Row num " << size << " match nlist " << nlist;
    }
    return nlist;
}

int64_t
MatchNbits(int64_t size, int64_t nbits) {
    if (size < (1 << nbits)) {
        // nbits is too large, adjust to a proper value
        LOG_KNOWHERE_WARNING_ << "nbits(" << nbits << ") is too large, adjust to a proper value";
        if (size >= (1 << 8)) {
            nbits = 8;
        } else if (size >= (1 << 4)) {
            nbits = 4;
        } else if (size >= (1 << 2)) {
            nbits = 2;
        } else {
            nbits = 1;
        }
        LOG_KNOWHERE_WARNING_ << "Row num " << size << " match nbits " << nbits;
    }
    return nbits;
}

template <typename T>
Status
IvfIndexNode<T>::Train(const DataSet& dataset, const Config& cfg) {
    const BaseConfig& base_cfg = static_cast<const IvfConfig&>(cfg);
    auto metric = Str2FaissMetricType(base_cfg.metric_type);
    if (!metric.has_value()) {
        return Status::invalid_metric_type;
    }
    decltype(this->qzr_) qzr = nullptr;
    decltype(this->index_) index = nullptr;
    auto dim = dataset.GetDim();
    try {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            const IvfFlatConfig& ivf_flat_cfg = static_cast<const IvfFlatConfig&>(cfg);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = new (std::nothrow) T(qzr, dim, MatchNlist(dataset.GetRows(), ivf_flat_cfg.nlist), metric.value());
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            const IvfPqConfig& ivf_pq_cfg = static_cast<const IvfPqConfig&>(cfg);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = new (std::nothrow) T(qzr, dim, MatchNlist(dataset.GetRows(), ivf_pq_cfg.nlist), ivf_pq_cfg.m,
                                         MatchNbits(dataset.GetRows(), ivf_pq_cfg.nbits), metric.value());
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            const IvfSqConfig& ivf_sq_cfg = static_cast<const IvfSqConfig&>(cfg);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = new (std::nothrow) T(qzr, dim, MatchNlist(dataset.GetRows(), ivf_sq_cfg.nlist),
                                         faiss::QuantizerType::QT_8bit, metric.value());
        }
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            const IvfBinConfig& ivf_bin_cfg = static_cast<const IvfBinConfig&>(cfg);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = new (std::nothrow) T(qzr, dim, MatchNlist(dataset.GetRows(), ivf_bin_cfg.nlist), metric.value());
        }
        if (qzr == nullptr || index == nullptr) {
            if (qzr) {
                delete qzr;
            }
            if (index) {
                delete index;
            }
            LOG_KNOWHERE_WARNING_ << "memory malloc error.";
            return Status::malloc_error;
        }
    } catch (std::exception& e) {
        if (qzr) {
            delete qzr;
        }
        if (index) {
            delete index;
        }
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return Status::faiss_inner_error;
    }

    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    try {
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            index->train(rows, (const uint8_t*)data);
        } else {
            index->train(rows, (const float*)data);
        }
    } catch (std::exception& e) {
        delete qzr;
        delete index;
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return Status::faiss_inner_error;
    }
    if (this->index_) {
        LOG_KNOWHERE_WARNING_ << "index not empty before train, delete old index";
        delete this->index_;
    }
    this->index_ = index;
    if (this->qzr_) {
        LOG_KNOWHERE_WARNING_ << "quantizer not empty before train, delete old quantizer.";
        delete this->qzr_;
    }
    this->qzr_ = qzr;
    return Status::success;
}

template <typename T>
Status
IvfIndexNode<T>::Add(const DataSet& dataset, const Config&) {
    if (!this->index_) {
        return Status::empty_index;
    }
    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    try {
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            index_->add_without_codes(rows, (const float*)data);
            auto raw_data = dataset.GetTensor();
            auto invlists = index_->invlists;
            auto d = index_->d;
            size_t nb = dataset.GetRows();
            index_->prefix_sum.resize(invlists->nlist);
            size_t curr_index = 0;

            auto ails = dynamic_cast<faiss::ArrayInvertedLists*>(invlists);
            index_->arranged_codes.resize(d * nb * sizeof(float));
            for (size_t i = 0; i < invlists->nlist; i++) {
                auto list_size = ails->ids[i].size();
                for (size_t j = 0; j < list_size; j++) {
                    memcpy(index_->arranged_codes.data() + d * (curr_index + j) * sizeof(float),
                           (uint8_t*)raw_data + d * ails->ids[i][j] * sizeof(float), d * sizeof(float));
                }
                index_->prefix_sum[i] = curr_index;
                curr_index += list_size;
            }
        } else if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            index_->add(rows, (const uint8_t*)data);
        } else {
            index_->add(rows, (const float*)data);
        }

    } catch (std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

template <typename T>
expected<DataSetPtr, Status>
IvfIndexNode<T>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_) {
        LOG_KNOWHERE_WARNING_ << "search on empty index";
        return unexpected(Status::empty_index);
    }
    if (!this->index_->is_trained) {
        LOG_KNOWHERE_WARNING_ << "index not trained";
        return unexpected(Status::index_not_trained);
    }
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
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            index_->search_without_codes_thread_safe(rows, (const float*)data, ivf_cfg.k, dis, ids, ivf_cfg.nprobe,
                                                     parallel_mode, max_codes, bitset);
        } else {
            index_->search_thread_safe(rows, (const float*)data, ivf_cfg.k, dis, ids, ivf_cfg.nprobe, parallel_mode,
                                       max_codes, bitset);
        }
    } catch (const std::exception& e) {
        delete[] ids;
        delete[] dis;
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return unexpected(Status::faiss_inner_error);
    }

    auto res = GenResultDataSet(rows, ivf_cfg.k, ids, dis);
    res->SetIsOwner(true);
    return res;
}

template <typename T>
expected<DataSetPtr, Status>
IvfIndexNode<T>::RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_) {
        LOG_KNOWHERE_WARNING_ << "range search on empty index.";
        return unexpected(Status::empty_index);
    }
    if (!this->index_->is_trained) {
        LOG_KNOWHERE_WARNING_ << "index not trained.";
        return unexpected(Status::index_not_trained);
    }

    auto nq = dataset.GetRows();
    auto xq = dataset.GetTensor();
    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);

    int parallel_mode = 0;
    if (ivf_cfg.nprobe > 1 && nq <= 4) {
        parallel_mode = 1;
    }

    float radius = ivf_cfg.radius;
    bool is_ip = (index_->metric_type == faiss::METRIC_INNER_PRODUCT);

    int64_t* ids = nullptr;
    float* distances = nullptr;
    size_t* lims = nullptr;

    try {
        size_t max_codes = 0;
        faiss::RangeSearchResult res(nq);
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            index_->range_search_without_codes_thread_safe(nq, (const float*)xq, radius, &res, ivf_cfg.nprobe,
                                                           parallel_mode, max_codes, bitset);
        } else {
            index_->range_search_thread_safe(nq, (const float*)xq, radius, &res, ivf_cfg.nprobe, parallel_mode,
                                             max_codes, bitset);
        }
        if (ivf_cfg.range_filter_exist) {
            GetRangeSearchResult(res, is_ip, nq, radius, ivf_cfg.range_filter, distances, ids, lims, bitset);
        } else {
            GetRangeSearchResult(res, is_ip, nq, radius, distances, ids, lims);
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return unexpected(Status::faiss_inner_error);
    }

    return GenResultDataSet(nq, ids, distances, lims);
}

template <>
expected<DataSetPtr, Status>
IvfIndexNode<faiss::IndexBinaryIVF>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_) {
        LOG_KNOWHERE_WARNING_ << "search on empty index";
        return unexpected(Status::empty_index);
    }
    if (!this->index_->is_trained) {
        LOG_KNOWHERE_WARNING_ << "index not trained";
        return unexpected(Status::index_not_trained);
    }
    auto rows = dataset.GetRows();
    auto data = dataset.GetTensor();
    auto ivf_bin_cfg = static_cast<const IvfBinConfig&>(cfg);

    int64_t* ids(new (std::nothrow) int64_t[rows * ivf_bin_cfg.k]);
    float* dis(new (std::nothrow) float[rows * ivf_bin_cfg.k]);
    auto i_dis = reinterpret_cast<int32_t*>(dis);
    try {
        index_->nprobe = ivf_bin_cfg.nprobe;
        index_->search(rows, (const uint8_t*)data, ivf_bin_cfg.k, i_dis, ids, bitset);
    } catch (const std::exception& e) {
        delete[] ids;
        delete[] dis;
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return unexpected(Status::faiss_inner_error);
    }

    if (index_->metric_type == faiss::METRIC_Hamming) {
        int64_t num = rows * ivf_bin_cfg.k;
        for (int64_t i = 0; i < num; i++) {
            dis[i] = static_cast<float>(i_dis[i]);
        }
    }
    return GenResultDataSet(rows, ivf_bin_cfg.k, ids, dis);
}

template <>
expected<DataSetPtr, Status>
IvfIndexNode<faiss::IndexBinaryIVF>::RangeSearch(const DataSet& dataset, const Config& cfg,
                                                 const BitsetView& bitset) const {
    if (!this->index_) {
        LOG_KNOWHERE_WARNING_ << "range search on empty index.";
        return unexpected(Status::empty_index);
    }
    if (!this->index_->is_trained) {
        LOG_KNOWHERE_WARNING_ << "index not trained.";
        return unexpected(Status::index_not_trained);
    }

    auto nq = dataset.GetRows();
    auto xq = dataset.GetTensor();
    auto ivf_bin_cfg = static_cast<const IvfBinConfig&>(cfg);

    float radius = ivf_bin_cfg.radius;

    int64_t* ids = nullptr;
    float* distances = nullptr;
    size_t* lims = nullptr;

    try {
        index_->nprobe = ivf_bin_cfg.nprobe;
        faiss::RangeSearchResult res(nq);
        index_->range_search(nq, (const uint8_t*)xq, radius, &res, bitset);
        if (ivf_bin_cfg.range_filter_exist) {
            GetRangeSearchResult(res, false, nq, radius, ivf_bin_cfg.range_filter, distances, ids, lims, bitset);
        } else {
            GetRangeSearchResult(res, false, nq, radius, distances, ids, lims);
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return unexpected(Status::faiss_inner_error);
    }

    return GenResultDataSet(nq, ids, distances, lims);
}

template <typename T>
expected<DataSetPtr, Status>
IvfIndexNode<T>::GetVectorByIds(const DataSet& dataset, const Config& cfg) const {
    if (!this->index_) {
        return unexpected(Status::empty_index);
    }
    if (!this->index_->is_trained) {
        return unexpected(Status::index_not_trained);
    }
    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    float* p_x(new (std::nothrow) float[dim * rows]);
    index_->make_direct_map(true);
    auto p_ids = dataset.GetIds();
    try {
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            assert(id >= 0 && id < index_->ntotal);
            if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
                index_->reconstruct_without_codes(id, p_x + i * dim);
            } else {
                index_->reconstruct(id, p_x + i * dim);
            }
        }
    } catch (const std::exception& e) {
        std::unique_ptr<float> p_x_auto_delete(p_x);
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return unexpected(Status::faiss_inner_error);
    }
    return GenResultDataSet(p_x);
}

template <>
expected<DataSetPtr, Status>
IvfIndexNode<faiss::IndexBinaryIVF>::GetVectorByIds(const DataSet& dataset, const Config& cfg) const {
    if (!this->index_) {
        return unexpected(Status::empty_index);
    }
    if (!this->index_->is_trained) {
        return unexpected(Status::index_not_trained);
    }
    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    uint8_t* p_x(new (std::nothrow) uint8_t[dim * rows / 8]);
    index_->make_direct_map(true);
    auto p_ids = dataset.GetIds();
    try {
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            assert(id >= 0 && id < index_->ntotal);
            index_->reconstruct(id, p_x + i * dim / 8);
        }
    } catch (const std::exception& e) {
        std::unique_ptr<uint8_t> p_x_auto_delete(p_x);
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return unexpected(Status::faiss_inner_error);
    }
    return GenResultDataSet(p_x);
}

template <>
expected<DataSetPtr, Status>
IvfIndexNode<faiss::IndexIVFFlat>::GetIndexMeta(const Config& config) const {
    if (!index_) {
        LOG_KNOWHERE_WARNING_ << "get index meta on empty index.";
        return unexpected(Status::empty_index);
    }

    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_);
    auto ivf_quantizer = dynamic_cast<faiss::IndexFlat*>(ivf_index->quantizer);

    int64_t dim = ivf_index->d;
    int64_t nlist = ivf_index->nlist;
    int64_t ntotal = ivf_index->ntotal;

    feder::ivfflat::IVFFlatMeta meta(nlist, dim, ntotal);
    std::unordered_set<int64_t> id_set;

    for (int32_t i = 0; i < nlist; i++) {
        // copy from IndexIVF::search_preassigned_without_codes
        std::unique_ptr<faiss::InvertedLists::ScopedIds> sids =
            std::make_unique<faiss::InvertedLists::ScopedIds>(index_->invlists, i);

        // node ids
        auto node_num = index_->invlists->list_size(i);
        auto node_id_codes = sids->get();

        // centroid vector
        auto centroid_vec = ivf_quantizer->get_xb() + i * dim;

        meta.AddCluster(i, node_id_codes, node_num, centroid_vec, dim);
    }

    Json json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataSet(json_meta.dump(), json_id_set.dump());
}

template <typename T>
Status
IvfIndexNode<T>::Serialize(BinarySet& binset) const {
    try {
        MemoryIOWriter writer;
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            faiss::write_index_binary(index_, &writer);
        } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            faiss::write_index_nm(index_, &writer);
        } else {
            faiss::write_index(index_, &writer);
        }
        std::shared_ptr<uint8_t[]> data(writer.data_);
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            binset.Append("BinaryIVF", data, writer.rp);
        } else {
            binset.Append("IVF", data, writer.rp);
        }
        return Status::success;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return Status::faiss_inner_error;
    }
}

template <typename T>
Status
IvfIndexNode<T>::Deserialize(const BinarySet& binset) {
    std::string name = "IVF";
    if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
        name = "BinaryIVF";
    }
    auto binary = binset.GetByName(name);

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();
    if (index_) {
        LOG_KNOWHERE_WARNING_ << "index not empty, delte old index.";
        delete index_;
    }
    try {
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            index_ = static_cast<T*>(faiss::read_index_binary(&reader));
        } else {
            index_ = static_cast<T*>(faiss::read_index(&reader));
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

template <>
Status
IvfIndexNode<faiss::IndexIVFFlat>::Deserialize(const BinarySet& binset) {
    std::string name = "IVF";
    auto binary = binset.GetByName(name);

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();
    if (index_) {
        LOG_KNOWHERE_WARNING_ << "index not empty, delte old index.";
        delete index_;
    }
    try {
        index_ = static_cast<faiss::IndexIVFFlat*>(faiss::read_index_nm(&reader));

        // Construct arranged data from original data
        auto binary = binset.GetByName(RAW_DATA);
        auto invlists = index_->invlists;
        auto d = index_->d;
        size_t nb = binary->size / invlists->code_size;
        index_->prefix_sum.resize(invlists->nlist);
        size_t curr_index = 0;

        auto ails = dynamic_cast<faiss::ArrayInvertedLists*>(invlists);
        index_->arranged_codes.resize(d * nb * sizeof(float));
        for (size_t i = 0; i < invlists->nlist; i++) {
            auto list_size = ails->ids[i].size();
            for (size_t j = 0; j < list_size; j++) {
                memcpy(index_->arranged_codes.data() + d * (curr_index + j) * sizeof(float),
                       binary->data.get() + d * ails->ids[i][j] * sizeof(float), d * sizeof(float));
            }
            index_->prefix_sum[i] = curr_index;
            curr_index += list_size;
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

KNOWHERE_REGISTER_GLOBAL(IVFBIN, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<IvfIndexNode<faiss::IndexBinaryIVF>>(object));
});

KNOWHERE_REGISTER_GLOBAL(BIN_IVF_FLAT, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<IvfIndexNode<faiss::IndexBinaryIVF>>(object));
});

KNOWHERE_REGISTER_GLOBAL(IVFFLAT, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<IvfIndexNode<faiss::IndexIVFFlat>>(object));
});
KNOWHERE_REGISTER_GLOBAL(IVF_FLAT, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<IvfIndexNode<faiss::IndexIVFFlat>>(object));
});
KNOWHERE_REGISTER_GLOBAL(IVFPQ, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<IvfIndexNode<faiss::IndexIVFPQ>>(object));
});
KNOWHERE_REGISTER_GLOBAL(IVF_PQ, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<IvfIndexNode<faiss::IndexIVFPQ>>(object));
});

KNOWHERE_REGISTER_GLOBAL(IVFSQ, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(
        std::make_unique<IvfIndexNode<faiss::IndexIVFScalarQuantizer>>(object));
});
KNOWHERE_REGISTER_GLOBAL(IVF_SQ8, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(
        std::make_unique<IvfIndexNode<faiss::IndexIVFScalarQuantizer>>(object));
});

}  // namespace knowhere
