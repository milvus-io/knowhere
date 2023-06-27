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

#include <omp.h>

#include "common/range_util.h"
#include "diskann/aux_utils.h"
#include "diskann/pq_flash_index.h"
#include "index/diskann/diskann_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/thread_pool.h"
#ifndef _WINDOWS
#include "diskann/linux_aligned_file_reader.h"
#else
#include "diskann/windows_aligned_file_reader.h"
#endif
#include "knowhere/factory.h"
#include "knowhere/feder/DiskANN.h"
#include "knowhere/file_manager.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename T>
class DiskANNIndexNode : public IndexNode {
    static_assert(std::is_same_v<T, float>, "DiskANN only support float");

 public:
    DiskANNIndexNode(const Object& object) : is_prepared_(false), dim_(-1), count_(-1) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<FileManager>>));
        auto diskann_index_pack = dynamic_cast<const Pack<std::shared_ptr<FileManager>>*>(&object);
        assert(diskann_index_pack != nullptr);
        file_manager_ = diskann_index_pack->GetPack();
    }

    Status
    Build(const DataSet& dataset, const Config& cfg) override;

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override;

    bool
    HasRawData(const std::string& metric_type) const override {
        return IsMetricType(metric_type, metric::L2) || IsMetricType(metric_type, metric::COSINE);
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override;

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_ERROR_ << "DiskANN doesn't support Serialize.";
        return Status::not_implemented;
    }

    Status
    Deserialize(const BinarySet& binset, const Config& cfg) override;

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        LOG_KNOWHERE_ERROR_ << "DiskANN doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<DiskANNConfig>();
    }

    Status
    SetFileManager(std::shared_ptr<FileManager> file_manager) {
        if (file_manager == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Malloc error, file_manager = nullptr.";
            return Status::malloc_error;
        }
        file_manager_ = file_manager;
        return Status::success;
    }

    int64_t
    Dim() const override {
        if (dim_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "Dim() function is not supported when index is not ready yet.";
            return 0;
        }
        return dim_.load();
    }

    int64_t
    Size() const override {
        LOG_KNOWHERE_ERROR_ << "Size() function has not been implemented yet.";
        return 0;
    }

    int64_t
    Count() const override {
        if (count_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "Count() function is not supported when index is not ready yet.";
            return 0;
        }
        return count_.load();
    }

    std::string
    Type() const override {
        return std::string(knowhere::IndexEnum::INDEX_DISKANN);
    }

 private:
    bool
    LoadFile(const std::string& filename) {
        if (!file_manager_->LoadFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    bool
    AddFile(const std::string& filename) {
        if (!file_manager_->AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    uint64_t
    GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim, const uint64_t max_degree);

    std::string index_prefix_;
    mutable std::mutex preparation_lock_;
    std::atomic_bool is_prepared_;
    std::shared_ptr<FileManager> file_manager_;
    std::unique_ptr<diskann::PQFlashIndex<T>> pq_flash_index_;
    std::atomic_int64_t dim_;
    std::atomic_int64_t count_;
    std::shared_ptr<ThreadPool> pool_;
};

}  // namespace knowhere

namespace knowhere {
namespace {
static constexpr float kCacheExpansionRate = 1.2;
static constexpr int kSearchListSizeMaxValue = 200;
static constexpr int64_t kGetVectorBatchSize = 32;
Status
TryDiskANNCall(std::function<void()>&& diskann_call) {
    try {
        diskann_call();
        return Status::success;
    } catch (const diskann::FileException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN File Exception: " << e.what();
        return Status::diskann_file_error;
    } catch (const diskann::ANNException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Exception: " << e.what();
        return Status::diskann_inner_error;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Other Exception: " << e.what();
        return Status::diskann_inner_error;
    }
}

std::vector<std::string>
GetNecessaryFilenames(const std::string& prefix, const bool need_norm, const bool use_sample_cache,
                      const bool use_sample_warmup) {
    std::vector<std::string> filenames;
    auto pq_pivots_filename = diskann::get_pq_pivots_filename(prefix);
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);

    filenames.push_back(pq_pivots_filename);
    filenames.push_back(diskann::get_pq_rearrangement_perm_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_chunk_offsets_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_centroid_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_compressed_filename(prefix));
    filenames.push_back(disk_index_filename);
    if (need_norm) {
        filenames.push_back(diskann::get_disk_index_max_base_norm_file(disk_index_filename));
    }
    if (use_sample_cache || use_sample_warmup) {
        filenames.push_back(diskann::get_sample_data_filename(prefix));
    }
    return filenames;
}

std::vector<std::string>
GetOptionalFilenames(const std::string& prefix) {
    std::vector<std::string> filenames;
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);
    filenames.push_back(diskann::get_disk_index_centroids_filename(disk_index_filename));
    filenames.push_back(diskann::get_disk_index_medoids_filename(disk_index_filename));
    filenames.push_back(diskann::get_cached_nodes_file(prefix));
    return filenames;
}

inline bool
AnyIndexFileExist(const std::string& index_prefix) {
    auto file_exist = [](std::vector<std::string> filenames) -> bool {
        for (auto& filename : filenames) {
            if (file_exists(filename)) {
                return true;
            }
        }
        return false;
    };
    return file_exist(GetNecessaryFilenames(index_prefix, diskann::INNER_PRODUCT, true, true)) ||
           file_exist(GetOptionalFilenames(index_prefix));
}

inline bool
CheckMetric(const std::string& diskann_metric) {
    if (diskann_metric != knowhere::metric::L2 && diskann_metric != knowhere::metric::IP &&
        diskann_metric != knowhere::metric::COSINE) {
        LOG_KNOWHERE_ERROR_ << "DiskANN currently only supports floating point data for Minimum Euclidean "
                               "distance(L2), Max Inner Product Search(IP) "
                               "and Minimum Cosine Search(COSINE)."
                            << std::endl;
        return false;
    } else {
        return true;
    }
}
}  // namespace

template <typename T>
Status
DiskANNIndexNode<T>::Build(const DataSet& dataset, const Config& cfg) {
    assert(file_manager_ != nullptr);
    std::lock_guard<std::mutex> lock(preparation_lock_);
    auto build_conf = static_cast<const DiskANNConfig&>(cfg);
    if (!CheckMetric(build_conf.metric_type.value())) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << build_conf.metric_type.value();
        return Status::invalid_metric_type;
    }
    if (AnyIndexFileExist(build_conf.index_prefix.value())) {
        LOG_KNOWHERE_ERROR_ << "This index prefix already has index files." << std::endl;
        return Status::diskann_file_error;
    }
    if (!LoadFile(build_conf.data_path.value())) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building." << std::endl;
        return Status::diskann_file_error;
    }
    auto& data_path = build_conf.data_path.value();
    index_prefix_ = build_conf.index_prefix.value();

    size_t count;
    size_t dim;
    diskann::get_bin_metadata(build_conf.data_path.value(), count, dim);
    count_.store(count);
    dim_.store(dim);

    bool need_norm = IsMetricType(build_conf.metric_type.value(), knowhere::metric::IP) ||
                     IsMetricType(build_conf.metric_type.value(), knowhere::metric::COSINE);
    auto diskann_metric = [m = build_conf.metric_type.value()] {
        if (IsMetricType(m, knowhere::metric::L2)) {
            return diskann::Metric::L2;
        } else if (IsMetricType(m, knowhere::metric::COSINE)) {
            return diskann::Metric::COSINE;
        } else {
            return diskann::Metric::INNER_PRODUCT;
        }
    }();
    auto num_nodes_to_cache =
        GetCachedNodeNum(build_conf.search_cache_budget_gb.value(), dim, build_conf.max_degree.value());
    diskann::BuildConfig diskann_internal_build_config{data_path,
                                                       index_prefix_,
                                                       diskann_metric,
                                                       static_cast<unsigned>(build_conf.max_degree.value()),
                                                       static_cast<unsigned>(build_conf.search_list_size.value()),
                                                       static_cast<double>(build_conf.pq_code_budget_gb.value()),
                                                       static_cast<double>(build_conf.build_dram_budget_gb.value()),
                                                       static_cast<uint32_t>(build_conf.disk_pq_dims.value()),
                                                       false,
                                                       build_conf.accelerate_build.value(),
                                                       static_cast<uint32_t>(num_nodes_to_cache)};
    RETURN_IF_ERROR(TryDiskANNCall([&]() {
        int res = diskann::build_disk_index<T>(diskann_internal_build_config);
        if (res != 0)
            throw diskann::ANNException("diskann::build_disk_index returned non-zero value: " + std::to_string(res),
                                        -1);
    }));

    // Add file to the file manager
    for (auto& filename : GetNecessaryFilenames(index_prefix_, need_norm, true, true)) {
        if (!AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::diskann_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        if (file_exists(filename) && !AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::diskann_file_error;
        }
    }

    is_prepared_.store(false);
    return Status::success;
}

template <typename T>
Status
DiskANNIndexNode<T>::Deserialize(const BinarySet& binset, const Config& cfg) {
    std::lock_guard<std::mutex> lock(preparation_lock_);
    auto prep_conf = static_cast<const DiskANNConfig&>(cfg);
    if (!CheckMetric(prep_conf.metric_type.value())) {
        return Status::invalid_metric_type;
    }
    if (is_prepared_.load()) {
        return Status::success;
    }
    index_prefix_ = prep_conf.index_prefix.value();
    bool is_ip = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP);
    bool need_norm = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP) ||
                     IsMetricType(prep_conf.metric_type.value(), knowhere::metric::COSINE);
    auto diskann_metric = [m = prep_conf.metric_type.value()] {
        if (IsMetricType(m, knowhere::metric::L2)) {
            return diskann::Metric::L2;
        } else if (IsMetricType(m, knowhere::metric::COSINE)) {
            return diskann::Metric::COSINE;
        } else {
            return diskann::Metric::INNER_PRODUCT;
        }
    }();

    // Load file from file manager.
    for (auto& filename : GetNecessaryFilenames(
             index_prefix_, need_norm, prep_conf.search_cache_budget_gb.value() > 0 && !prep_conf.use_bfs_cache.value(),
             prep_conf.warm_up.value())) {
        if (!LoadFile(filename)) {
            return Status::diskann_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        auto is_exist_op = file_manager_->IsExisted(filename);
        if (!is_exist_op.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
            return Status::diskann_file_error;
        }
        if (is_exist_op.value() && !LoadFile(filename)) {
            return Status::diskann_file_error;
        }
    }

    // set thread pool
    pool_ = ThreadPool::GetGlobalThreadPool();

    // load diskann pq code and meta info
    std::shared_ptr<AlignedFileReader> reader = nullptr;

    reader.reset(new LinuxAlignedFileReader());

    pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<T>>(reader, diskann_metric);
    auto disk_ann_call = [&]() {
        int res = pq_flash_index_->load(pool_->size(), index_prefix_.c_str());
        if (res != 0) {
            throw diskann::ANNException("pq_flash_index_->load returned non-zero value: " + std::to_string(res), -1);
        }
    };
    if (TryDiskANNCall(disk_ann_call) != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Failed to load DiskANN.";
        return Status::diskann_inner_error;
    }

    count_.store(pq_flash_index_->get_num_points());
    // DiskANN will add one more dim for IP type.
    if (is_ip) {
        dim_.store(pq_flash_index_->get_data_dim() - 1);
    } else {
        dim_.store(pq_flash_index_->get_data_dim());
    }

    std::string warmup_query_file = diskann::get_sample_data_filename(index_prefix_);
    // load cache
    auto cached_nodes_file = diskann::get_cached_nodes_file(index_prefix_);
    std::vector<uint32_t> node_list;
    if (file_exists(cached_nodes_file)) {
        LOG_KNOWHERE_INFO_ << "Reading cached nodes from file.";
        size_t num_nodes, nodes_id_dim;
        uint32_t* cached_nodes_ids = nullptr;
        diskann::load_bin<uint32_t>(cached_nodes_file, cached_nodes_ids, num_nodes, nodes_id_dim);
        node_list.assign(cached_nodes_ids, cached_nodes_ids + num_nodes);
        if (cached_nodes_ids != nullptr) {
            delete[] cached_nodes_ids;
        }
    } else {
        auto num_nodes_to_cache = GetCachedNodeNum(prep_conf.search_cache_budget_gb.value(),
                                                   pq_flash_index_->get_data_dim(), pq_flash_index_->get_max_degree());
        if (num_nodes_to_cache > pq_flash_index_->get_num_points() / 3) {
            LOG_KNOWHERE_ERROR_ << "Failed to generate cache, num_nodes_to_cache(" << num_nodes_to_cache
                                << ") is larger than 1/3 of the total data number.";
            return Status::invalid_args;
        }
        if (num_nodes_to_cache > 0) {
            LOG_KNOWHERE_INFO_ << "Caching " << num_nodes_to_cache << " sample nodes around medoid(s).";
            if (prep_conf.use_bfs_cache.value()) {
                LOG_KNOWHERE_INFO_ << "Use bfs to generate cache list";
                if (TryDiskANNCall([&]() { pq_flash_index_->cache_bfs_levels(num_nodes_to_cache, node_list); }) !=
                    Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate bfs cache for DiskANN.";
                    return Status::diskann_inner_error;
                }
            } else {
                LOG_KNOWHERE_INFO_ << "Use sample_queries to generate cache list";
                if (TryDiskANNCall([&]() {
                        pq_flash_index_->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6,
                                                                                 num_nodes_to_cache, node_list);
                    }) != Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate cache from sample queries for DiskANN.";
                    return Status::diskann_inner_error;
                }
            }
        }
        LOG_KNOWHERE_INFO_ << "End of preparing diskann index.";
    }

    if (node_list.size() > 0) {
        if (TryDiskANNCall([&]() { pq_flash_index_->load_cache_list(node_list); }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load cache for DiskANN.";
            return Status::diskann_inner_error;
        }
    }

    // warmup
    if (prep_conf.warm_up.value()) {
        LOG_KNOWHERE_INFO_ << "Warming up.";
        uint64_t warmup_L = 20;
        uint64_t warmup_num = 0;
        uint64_t warmup_dim = 0;
        uint64_t warmup_aligned_dim = 0;
        T* warmup = nullptr;
        if (TryDiskANNCall([&]() {
                diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
            }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for DiskANN.";
            return Status::diskann_file_error;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

        bool all_searches_are_good = true;

        std::vector<std::future<void>> futures;
        futures.reserve(warmup_num);
        for (_s64 i = 0; i < (int64_t)warmup_num; ++i) {
            futures.push_back(pool_->push([&, index = i]() {
                pq_flash_index_->cached_beam_search(warmup + (index * warmup_aligned_dim), 1, warmup_L,
                                                    warmup_result_ids_64.data() + (index * 1),
                                                    warmup_result_dists.data() + (index * 1), 4);
            }));
        }
        for (auto& future : futures) {
            if (TryDiskANNCall([&]() { future.get(); }) != Status::success) {
                all_searches_are_good = false;
            }
        }
        if (warmup != nullptr) {
            diskann::aligned_free(warmup);
        }

        if (!all_searches_are_good) {
            LOG_KNOWHERE_ERROR_ << "Failed to do search on warmup file for DiskANN.";
            return Status::diskann_inner_error;
        }
    }

    is_prepared_.store(true);
    return Status::success;
}

template <typename T>
expected<DataSetPtr>
DiskANNIndexNode<T>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return Status::empty_index;
    }

    auto search_conf = static_cast<const DiskANNConfig&>(cfg);
    if (!CheckMetric(search_conf.metric_type.value())) {
        return Status::invalid_metric_type;
    }
    auto max_search_list_size = std::max(kSearchListSizeMaxValue, search_conf.k.value() * 10);
    if (search_conf.search_list_size.value() > max_search_list_size ||
        search_conf.search_list_size.value() < search_conf.k.value()) {
        LOG_KNOWHERE_ERROR_ << "search_list_size should be in range: [topk, max(200, topk * 10)]";
        return Status::invalid_args;
    }
    auto k = static_cast<uint64_t>(search_conf.k.value());
    auto lsearch = static_cast<uint64_t>(search_conf.search_list_size.value());
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value());
    auto filter_ratio = static_cast<float>(search_conf.filter_threshold.value());
    auto for_tuning = static_cast<bool>(search_conf.for_tuning.value());

    auto nq = dataset.GetRows();
    auto dim = dataset.GetDim();
    auto xq = static_cast<const T*>(dataset.GetTensor());

    feder::diskann::FederResultUniq feder_result;
    if (search_conf.trace_visit.value()) {
        if (nq != 1) {
            return Status::invalid_args;
        }
        feder_result = std::make_unique<feder::diskann::FederResult>();
        feder_result->visit_info_.SetQueryConfig(search_conf.k.value(), search_conf.beamwidth.value(),
                                                 search_conf.search_list_size.value());
    }

    auto p_id = new int64_t[k * nq];
    auto p_dist = new float[k * nq];

    bool all_searches_are_good = true;
    std::vector<std::future<void>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.push_back(pool_->push([&, index = row]() {
            pq_flash_index_->cached_beam_search(xq + (index * dim), k, lsearch, p_id + (index * k),
                                                p_dist + (index * k), beamwidth, false, nullptr, feder_result, bitset,
                                                filter_ratio, for_tuning);
        }));
    }
    for (auto& future : futures) {
        if (TryDiskANNCall([&]() { future.get(); }) != Status::success) {
            all_searches_are_good = false;
        }
    }

    if (!all_searches_are_good) {
        return Status::diskann_inner_error;
    }

    auto res = GenResultDataSet(nq, k, p_id, p_dist);

    // set visit_info json string into result dataset
    if (feder_result != nullptr) {
        Json json_visit_info, json_id_set;
        nlohmann::to_json(json_visit_info, feder_result->visit_info_);
        nlohmann::to_json(json_id_set, feder_result->id_set_);
        res->SetJsonInfo(json_visit_info.dump());
        res->SetJsonIdSet(json_id_set.dump());
    }
    return res;
}

template <typename T>
expected<DataSetPtr>
DiskANNIndexNode<T>::RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return Status::empty_index;
    }

    auto search_conf = static_cast<const DiskANNConfig&>(cfg);
    if (!CheckMetric(search_conf.metric_type.value())) {
        return Status::invalid_metric_type;
    }
    if (search_conf.min_k.value() > search_conf.max_k.value()) {
        LOG_KNOWHERE_ERROR_ << "min_k should be smaller than max_k";
        return Status::invalid_args;
    }
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value());
    auto min_k = static_cast<uint64_t>(search_conf.min_k.value());
    auto max_k = static_cast<uint64_t>(search_conf.max_k.value());
    auto search_list_and_k_ratio = search_conf.search_list_and_k_ratio.value();

    auto radius = search_conf.radius.value();
    bool is_ip = (pq_flash_index_->get_metric() == diskann::Metric::INNER_PRODUCT);

    auto dim = dataset.GetDim();
    auto nq = dataset.GetRows();
    auto xq = static_cast<const T*>(dataset.GetTensor());

    int64_t* p_id = nullptr;
    float* p_dist = nullptr;
    size_t* p_lims = nullptr;

    std::vector<std::vector<int64_t>> result_id_array(nq);
    std::vector<std::vector<float>> result_dist_array(nq);

    std::vector<std::future<void>> futures;
    futures.reserve(nq);
    bool all_searches_are_good = true;
    for (int64_t row = 0; row < nq; ++row) {
        futures.push_back(pool_->push([&, index = row]() {
            std::vector<int64_t> indices;
            std::vector<float> distances;
            pq_flash_index_->range_search(xq + (index * dim), radius, min_k, max_k, result_id_array[index],
                                          result_dist_array[index], beamwidth, search_list_and_k_ratio, bitset);
            // filter range search result
            if (search_conf.range_filter.value() != defaultRangeFilter) {
                FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                search_conf.range_filter.value());
            }
        }));
    }
    for (auto& future : futures) {
        if (TryDiskANNCall([&]() { future.get(); }) != Status::success) {
            all_searches_are_good = false;
        }
    }
    if (!all_searches_are_good) {
        return Status::diskann_inner_error;
    }

    GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, search_conf.range_filter.value(),
                         p_dist, p_id, p_lims);
    return GenResultDataSet(nq, p_id, p_dist, p_lims);
}

template <typename T>
expected<DataSetPtr>
DiskANNIndexNode<T>::GetVectorByIds(const DataSet& dataset) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return Status::empty_index;
    }

    auto dim = Dim();
    auto rows = dataset.GetRows();
    auto ids = dataset.GetIds();
    float* data = new float[dim * rows];
    if (data == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Failed to allocate memory for data.";
        return Status::malloc_error;
    }

    auto batch_num = (rows + kGetVectorBatchSize - 1) / kGetVectorBatchSize;
    std::vector<std::future<void>> futures;
    futures.reserve(batch_num);
    bool all_good = true;
    for (int64_t i = 0; i < batch_num; ++i) {
        auto idx = i * kGetVectorBatchSize;
        auto len = std::min(kGetVectorBatchSize, rows - idx);
        futures.push_back(pool_->push([=]() { pq_flash_index_->get_vector_by_ids(ids + idx, len, data + idx * dim); }));
    }
    for (auto& future : futures) {
        if (TryDiskANNCall([&]() { future.get(); }) != Status::success) {
            all_good = false;
        }
    }
    if (!all_good) {
        delete[] data;
        return Status::diskann_inner_error;
    }
    return GenResultDataSet(rows, dim, data);
}

template <typename T>
expected<DataSetPtr>
DiskANNIndexNode<T>::GetIndexMeta(const Config& cfg) const {
    std::vector<int64_t> entry_points;
    for (size_t i = 0; i < pq_flash_index_->get_num_medoids(); i++) {
        entry_points.push_back(pq_flash_index_->get_medoids()[i]);
    }
    auto diskann_conf = static_cast<const DiskANNConfig&>(cfg);
    feder::diskann::DiskANNMeta meta(diskann_conf.data_path.value(), diskann_conf.max_degree.value(),
                                     diskann_conf.search_list_size.value(), diskann_conf.pq_code_budget_gb.value(),
                                     diskann_conf.build_dram_budget_gb.value(), diskann_conf.disk_pq_dims.value(),
                                     diskann_conf.accelerate_build.value(), Count(), entry_points);
    std::unordered_set<int64_t> id_set(entry_points.begin(), entry_points.end());

    Json json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataSet(json_meta.dump(), json_id_set.dump());
}

template <typename T>
uint64_t
DiskANNIndexNode<T>::GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim,
                                      const uint64_t max_degree) {
    uint32_t one_cached_node_budget = (max_degree + 1) * sizeof(unsigned) + sizeof(T) * data_dim;
    auto num_nodes_to_cache =
        static_cast<uint64_t>(1024 * 1024 * 1024 * cache_dram_budget) / (one_cached_node_budget * kCacheExpansionRate);
    return num_nodes_to_cache;
}

KNOWHERE_REGISTER_GLOBAL(DISKANN, [](const Object& object) { return Index<DiskANNIndexNode<float>>::Create(object); });
}  // namespace knowhere
