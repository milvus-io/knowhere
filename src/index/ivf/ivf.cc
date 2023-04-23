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
#include "knowhere/utils.h"

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
    IvfIndexNode(const Object& object) : index_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexIVFFlat>::value || std::is_same<T, faiss::IndexIVFFlatCC>::value ||
                          std::is_same<T, faiss::IndexIVFPQ>::value ||
                          std::is_same<T, faiss::IndexIVFScalarQuantizer>::value ||
                          std::is_same<T, faiss::IndexBinaryIVF>::value,
                      "not support");
        pool_ = ThreadPool::GetGlobalThreadPool();
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
    bool
    HasRawData(const std::string& metric_type) const override {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            return true;
        }
        if constexpr (std::is_same<faiss::IndexIVFFlatCC, T>::value) {
            return false;
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            return false;
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            return false;
        }
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            return true;
        }
    }
    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }
    Status
    Serialize(BinarySet& binset) const override;
    Status
    Deserialize(const BinarySet& binset) override;
    Status
    DeserializeFromFile(const std::string& filename, const LoadConfig& config) override;
    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            return std::make_unique<IvfFlatConfig>();
        }
        if constexpr (std::is_same<faiss::IndexIVFFlatCC, T>::value) {
            return std::make_unique<IvfFlatCcConfig>();
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
        if constexpr (std::is_same<T, faiss::IndexIVFFlatCC>::value) {
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
        if constexpr (std::is_same<T, faiss::IndexIVFFlatCC>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC;
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

 private:
    std::unique_ptr<T> index_;
    std::shared_ptr<ThreadPool> pool_;
};

}  // namespace knowhere

namespace knowhere {

template <typename T>
Status
IvfIndexNode<T>::Build(const DataSet& dataset, const Config& cfg) {
    const BaseConfig& base_cfg = static_cast<const IvfConfig&>(cfg);
    auto build_thread_num = base_cfg.get_build_thread_num();
    ThreadPool::ScopedOmpSetter setter(build_thread_num);
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
    auto build_thread_num = base_cfg.get_build_thread_num();
    ThreadPool::ScopedOmpSetter setter(build_thread_num);

    // do normalize for COSINE metric type
    if (IsMetricType(base_cfg.metric_type, knowhere::metric::COSINE)) {
        Normalize(dataset);
    }

    auto metric = Str2FaissMetricType(base_cfg.metric_type);
    if (!metric.has_value()) {
        return Status::invalid_metric_type;
    }
    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    auto data = dataset.GetTensor();

    typename QuantizerT<T>::type* qzr = nullptr;
    std::unique_ptr<T> index;
    try {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            const IvfFlatConfig& ivf_flat_cfg = static_cast<const IvfFlatConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_flat_cfg.nlist);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = std::make_unique<faiss::IndexIVFFlat>(qzr, dim, nlist, metric.value());
            index->own_fields = true;
            index->train(rows, (const float*)data);
        }
        if constexpr (std::is_same<faiss::IndexIVFFlatCC, T>::value) {
            const IvfFlatCcConfig& ivf_flat_cc_cfg = static_cast<const IvfFlatCcConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_flat_cc_cfg.nlist);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = std::make_unique<faiss::IndexIVFFlatCC>(qzr, dim, nlist, ivf_flat_cc_cfg.ssize, metric.value());
            index->own_fields = true;
            index->train(rows, (const float*)data);
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            const IvfPqConfig& ivf_pq_cfg = static_cast<const IvfPqConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_pq_cfg.nlist);
            auto nbits = MatchNbits(rows, ivf_pq_cfg.nbits);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = std::make_unique<faiss::IndexIVFPQ>(qzr, dim, nlist, ivf_pq_cfg.m, nbits, metric.value());
            index->own_fields = true;
            index->train(rows, (const float*)data);
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            const IvfSqConfig& ivf_sq_cfg = static_cast<const IvfSqConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_sq_cfg.nlist);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = std::make_unique<faiss::IndexIVFScalarQuantizer>(qzr, dim, nlist, faiss::QuantizerType::QT_8bit,
                                                                     metric.value());
            index->own_fields = true;
            index->train(rows, (const float*)data);
        }
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            const IvfBinConfig& ivf_bin_cfg = static_cast<const IvfBinConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_bin_cfg.nlist);
            qzr = new (std::nothrow) typename QuantizerT<T>::type(dim, metric.value());
            index = std::make_unique<faiss::IndexBinaryIVF>(qzr, dim, nlist, metric.value());
            index->own_fields = true;
            index->train(rows, (const uint8_t*)data);
        }
    } catch (std::exception& e) {
        if (qzr) {
            delete qzr;
        }
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    index_ = std::move(index);

    return Status::success;
}

template <typename T>
Status
IvfIndexNode<T>::Add(const DataSet& dataset, const Config& cfg) {
    if (!this->index_) {
        return Status::empty_index;
    }
    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    const BaseConfig& base_cfg = static_cast<const IvfConfig&>(cfg);
    auto build_thread_num = base_cfg.get_build_thread_num();
    ThreadPool::ScopedOmpSetter setter(build_thread_num);
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
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
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

    auto dim = dataset.GetDim();
    auto rows = dataset.GetRows();
    auto data = dataset.GetTensor();

    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);

    // do normalize for COSINE metric type
    if (IsMetricType(ivf_cfg.metric_type, knowhere::metric::COSINE)) {
        Normalize(dataset);
    }

    auto k = ivf_cfg.k;
    auto nprobe = ivf_cfg.nprobe;

    int parallel_mode = 0;
    if (nprobe > 1 && rows <= 4) {
        parallel_mode = 1;
    }
    int64_t* ids(new (std::nothrow) int64_t[rows * k]);
    float* distances(new (std::nothrow) float[rows * k]);
    int32_t* i_distances = reinterpret_cast<int32_t*>(distances);
    try {
        size_t max_codes = 0;
        std::vector<std::future<void>> futs;
        futs.reserve(rows);
        for (int i = 0; i < rows; ++i) {
            futs.push_back(pool_->push([&, index = i] {
                ThreadPool::ScopedOmpSetter setter(1);
                auto offset = k * index;
                if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
                    auto cur_data = (const uint8_t*)data + index * dim / 8;
                    index_->search_thread_safe(1, cur_data, k, i_distances + offset, ids + offset, nprobe, bitset);
                    if (index_->metric_type == faiss::METRIC_Hamming) {
                        for (int64_t i = 0; i < k; i++) {
                            distances[i + offset] = static_cast<float>(i_distances[i + offset]);
                        }
                    }
                } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
                    auto cur_data = (const float*)data + index * dim;
                    index_->search_without_codes_thread_safe(1, cur_data, k, distances + offset, ids + offset, nprobe,
                                                             parallel_mode, max_codes, bitset);
                } else {
                    auto cur_data = (const float*)data + index * dim;
                    index_->search_thread_safe(1, cur_data, k, distances + offset, ids + offset, nprobe, parallel_mode,
                                               max_codes, bitset);
                }
            }));
        }
        for (auto& fut : futs) {
            fut.get();
        }
    } catch (const std::exception& e) {
        delete[] ids;
        delete[] distances;
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return unexpected(Status::faiss_inner_error);
    }

    auto res = GenResultDataSet(rows, ivf_cfg.k, ids, distances);
    res->SetIsOwner(true);
    return res;
}

template <typename T>
expected<DataSetPtr, Status>
IvfIndexNode<T>::RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_) {
        LOG_KNOWHERE_WARNING_ << "range search on empty index";
        return unexpected(Status::empty_index);
    }
    if (!this->index_->is_trained) {
        LOG_KNOWHERE_WARNING_ << "index not trained";
        return unexpected(Status::index_not_trained);
    }

    auto nq = dataset.GetRows();
    auto xq = dataset.GetTensor();
    auto dim = dataset.GetDim();

    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);

    // do normalize for COSINE metric type
    if (IsMetricType(ivf_cfg.metric_type, knowhere::metric::COSINE)) {
        Normalize(dataset);
    }

    auto nprobe = ivf_cfg.nprobe;

    int parallel_mode = 0;
    if (nprobe > 1 && nq <= 4) {
        parallel_mode = 1;
    }

    float radius = ivf_cfg.radius;
    float range_filter = ivf_cfg.range_filter;
    bool is_ip = (index_->metric_type == faiss::METRIC_INNER_PRODUCT);

    int64_t* ids = nullptr;
    float* distances = nullptr;
    size_t* lims = nullptr;

    std::vector<std::vector<int64_t>> result_id_array(nq);
    std::vector<std::vector<float>> result_dist_array(nq);
    std::vector<size_t> result_size(nq);
    std::vector<size_t> result_lims(nq + 1);

    try {
        size_t max_codes = 0;
        std::vector<std::future<void>> futs;
        futs.reserve(nq);
        for (int i = 0; i < nq; ++i) {
            futs.push_back(pool_->push([&, index = i] {
                ThreadPool::ScopedOmpSetter setter(1);
                faiss::RangeSearchResult res(1);
                if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
                    auto cur_data = (const uint8_t*)xq + index * dim / 8;
                    index_->range_search_thread_safe(1, cur_data, radius, &res, nprobe, bitset);
                } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
                    auto cur_data = (const float*)xq + index * dim;
                    index_->range_search_without_codes_thread_safe(1, cur_data, radius, &res, nprobe, parallel_mode,
                                                                   max_codes, bitset);
                } else {
                    auto cur_data = (const float*)xq + index * dim;
                    index_->range_search_thread_safe(1, cur_data, radius, &res, nprobe, parallel_mode, max_codes,
                                                     bitset);
                }
                auto elem_cnt = res.lims[1];
                result_dist_array[index].resize(elem_cnt);
                result_id_array[index].resize(elem_cnt);
                result_size[index] = elem_cnt;
                for (size_t j = 0; j < elem_cnt; j++) {
                    result_dist_array[index][j] = res.distances[j];
                    result_id_array[index][j] = res.labels[j];
                }
                if (range_filter != defaultRangeFilter) {
                    FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                    range_filter);
                }
            }));
        }
        for (auto& fut : futs) {
            fut.get();
        }
        GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, range_filter, distances, ids, lims);
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
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
    auto dim = Dim();
    auto rows = dataset.GetRows();
    auto ids = dataset.GetIds();

    if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
        uint8_t* data = nullptr;
        try {
            data = new uint8_t[dim * rows / 8];
            index_->make_direct_map(true);
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < index_->ntotal);
                index_->reconstruct(id, data + i * dim / 8);
            }
            return GenResultDataSet(rows, dim, data);
        } catch (const std::exception& e) {
            std::unique_ptr<uint8_t[]> auto_del(data);
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return unexpected(Status::faiss_inner_error);
        }
    } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
        float* data = nullptr;
        try {
            data = new float[dim * rows];
            index_->make_direct_map(true);
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < index_->ntotal);
                index_->reconstruct_without_codes(id, data + i * dim);
            }
            return GenResultDataSet(rows, dim, data);
        } catch (const std::exception& e) {
            std::unique_ptr<float[]> auto_del(data);
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return unexpected(Status::faiss_inner_error);
        }
    } else if constexpr (std::is_same<T, faiss::IndexIVFFlatCC>::value) {
        float* data = nullptr;
        try {
            data = new float[dim * rows];
            index_->make_direct_map(true);
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < index_->ntotal);
                index_->reconstruct(id, data + i * dim);
            }
            return GenResultDataSet(rows, dim, data);
        } catch (const std::exception& e) {
            std::unique_ptr<float[]> auto_del(data);
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return unexpected(Status::faiss_inner_error);
        }
    } else {
        return unexpected(Status::not_implemented);
    }
}

template <>
expected<DataSetPtr, Status>
IvfIndexNode<faiss::IndexIVFFlat>::GetIndexMeta(const Config& config) const {
    if (!index_) {
        LOG_KNOWHERE_WARNING_ << "get index meta on empty index";
        return unexpected(Status::empty_index);
    }

    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
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
            faiss::write_index_binary(index_.get(), &writer);
        } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            faiss::write_index_nm(index_.get(), &writer);
        } else {
            faiss::write_index(index_.get(), &writer);
        }
        std::shared_ptr<uint8_t[]> data(writer.data_);
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            binset.Append("BIN_IVF", data, writer.rp);
        } else {
            binset.Append("IVF", data, writer.rp);
        }
        return Status::success;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
}

template <typename T>
Status
IvfIndexNode<T>::Deserialize(const BinarySet& binset) {
    std::string name = "IVF";
    if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
        name = "BIN_IVF";
    }
    auto binary = binset.GetByName(name);

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();
    try {
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            index_.reset(static_cast<T*>(faiss::read_index_binary(&reader)));
        } else {
            index_.reset(static_cast<T*>(faiss::read_index(&reader)));
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

template <typename T>
Status
IvfIndexNode<T>::DeserializeFromFile(const std::string& filename, const LoadConfig& config) {
    int io_flags = 0;
    if (config.enable_mmap) {
        io_flags |= faiss::IO_FLAG_MMAP;
    }
    try {
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            index_.reset(static_cast<T*>(faiss::read_index_binary(filename.data(), io_flags)));
        } else {
            index_.reset(static_cast<T*>(faiss::read_index(filename.data(), io_flags)));
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
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
    try {
        index_.reset(static_cast<faiss::IndexIVFFlat*>(faiss::read_index_nm(&reader)));

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
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

KNOWHERE_REGISTER_GLOBAL(IVFBIN, [](const Object& object) {
    return Index<IvfIndexNode<faiss::IndexBinaryIVF>>::Create(object);
});

KNOWHERE_REGISTER_GLOBAL(BIN_IVF_FLAT, [](const Object& object) {
    return Index<IvfIndexNode<faiss::IndexBinaryIVF>>::Create(object);
});

KNOWHERE_REGISTER_GLOBAL(IVFFLAT,
                         [](const Object& object) { return Index<IvfIndexNode<faiss::IndexIVFFlat>>::Create(object); });
KNOWHERE_REGISTER_GLOBAL(IVF_FLAT,
                         [](const Object& object) { return Index<IvfIndexNode<faiss::IndexIVFFlat>>::Create(object); });
KNOWHERE_REGISTER_GLOBAL(IVFFLATCC, [](const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFFlatCC>>::Create(object);
});
KNOWHERE_REGISTER_GLOBAL(IVF_FLAT_CC, [](const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFFlatCC>>::Create(object);
});
KNOWHERE_REGISTER_GLOBAL(IVFPQ,
                         [](const Object& object) { return Index<IvfIndexNode<faiss::IndexIVFPQ>>::Create(object); });
KNOWHERE_REGISTER_GLOBAL(IVF_PQ,
                         [](const Object& object) { return Index<IvfIndexNode<faiss::IndexIVFPQ>>::Create(object); });

KNOWHERE_REGISTER_GLOBAL(IVFSQ, [](const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFScalarQuantizer>>::Create(object);
});
KNOWHERE_REGISTER_GLOBAL(IVF_SQ8, [](const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFScalarQuantizer>>::Create(object);
});

}  // namespace knowhere
