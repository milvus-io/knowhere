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

namespace knowhere {

template <typename T>
class DiskANNIndexNode : public IndexNode {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "DiskANN only support float, int8 and uint8");

 public:
    DiskANNIndexNode(const Object& object) : is_prepared_(false), dim_(-1), count_(-1) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<FileManager>>));
        auto diskann_index_pack = dynamic_cast<const Pack<std::shared_ptr<FileManager>>*>(&object);
        assert(diskann_index_pack != nullptr);
        file_manager_ = diskann_index_pack->GetPack();
    }

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) {
        return Add(dataset, cfg);
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) {
        return Status::success;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override;

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;

    virtual expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        LOG_KNOWHERE_ERROR_ << "DiskANN doesn't support GetVectorById.";
        return unexpected(Status::not_implemented);
    }

    virtual expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override;

    virtual Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_ERROR_ << "DiskANN doesn't support Serialize.";
        return Status::not_implemented;
    }

    virtual Status
    Deserialize(const BinarySet& binset) override {
        LOG_KNOWHERE_ERROR_ << "DiskANN doesn't support Deserialization.";
        return Status::not_implemented;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<DiskANNConfig>();
    }

    Status
    SetFileManager(std::shared_ptr<FileManager> file_manager) {
        if (file_manager == nullptr) {
            return Status::malloc_error;
        }
        file_manager_ = file_manager;
        return Status::success;
    }

    virtual int64_t
    Dim() const override {
        if (dim_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "index is not ready yet.";
            return 0;
        }
        return dim_.load();
    }

    virtual int64_t
    Size() const override {
        LOG_KNOWHERE_ERROR_ << "Size() function has not been implemented yet.";
        return 0;
    }

    virtual int64_t
    Count() const override {
        if (count_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "index is not ready yet.";
            return 0;
        }
        return count_.load();
    }

    virtual std::string
    Type() const override {
        if (std::is_same_v<T, float>) {
            return std::string(knowhere::IndexEnum::INDEX_DISKANN) + "FLOAT";
        } else if (std::is_same_v<T, uint8_t>) {
            return std::string(knowhere::IndexEnum::INDEX_DISKANN) + "UINT8";
        } else if (std::is_same_v<T, int8_t>) {
            return std::string(knowhere::IndexEnum::INDEX_DISKANN) + "INT8";
        }
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

    bool
    Prepare(const Config& cfg);

    uint64_t
    GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim, const uint64_t max_degree);

    std::string index_prefix_;
    mutable std::mutex preparation_lock_;
    std::atomic_bool is_prepared_;
    int32_t num_threads_;
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
static constexpr uint32_t kLinuxAioMaxnrLimit = 65536;
static constexpr int kSearchListSizeMaxValue = 200;
template <typename T>
expected<T, Status>
TryDiskANNCall(std::function<T()>&& diskann_call) {
    try {
        return diskann_call();
    } catch (const diskann::FileException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN File Exception: " << e.what();
        return unexpected(Status::diskann_file_error);
    } catch (const diskann::ANNException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Exception: " << e.what();
        return unexpected(Status::diskann_inner_error);
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Other Exception: " << e.what();
        return unexpected(Status::diskann_inner_error);
    }
}

std::vector<std::string>
GetNecessaryFilenames(const std::string& prefix, const bool is_inner_product, const bool use_sample_cache,
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
    if (is_inner_product) {
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
    return filenames;
}

inline bool
AnyIndexFileExist(const std::string& index_prefix) {
    auto file_exist = [&index_prefix](std::vector<std::string> filenames) -> bool {
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

inline Status
CheckAddParams(const DiskANNConfig& diskann_cfg) {
    if (diskann_cfg.metric_type != knowhere::metric::L2 || diskann_cfg.metric_type != knowhere::metric::IP) {
        LOG_KNOWHERE_ERROR_ << "DiskANN currently only supports floating point data for Max Inner Product Search. "
                            << std::endl;
        return Status::invalid_metric_type;
    }
    if (AnyIndexFileExist(diskann_cfg.index_prefix)) {
        LOG_KNOWHERE_ERROR_ << "This index prefix already has index files." << std::endl;
        return Status::diskann_file_error;
    }
    return Status::success;
}

inline Status
CheckSearchParams(const DiskANNConfig& diskann_cfg) {
    auto max_search_list_size = std::max(kSearchListSizeMaxValue, diskann_cfg.k * 10);
    if (diskann_cfg.search_list_size > max_search_list_size || diskann_cfg.search_list_size < diskann_cfg.k) {
        LOG_KNOWHERE_ERROR_ << "search_list_size should be in range: [topk, max(200, topk * 10)]";
        return Status::invalid_args;
    }
    return Status::success;
}

inline Status
CheckRangeSearchParams(const DiskANNConfig& diskann_cfg) {
    if (diskann_cfg.metric_type == knowhere::metric::L2) {
        if (diskann_cfg.min_k > diskann_cfg.max_k || diskann_cfg.radius_low_bound > diskann_cfg.radius_high_bound) {
            LOG_KNOWHERE_ERROR_ << "min_k should be smaller than max_k, radius_low_bound should be smaller than the "
                                   "radius_high_bound in metric L2";
            return Status::invalid_args;
        }
    } else if (diskann_cfg.metric_type == knowhere::metric::IP) {
        if (diskann_cfg.min_k < diskann_cfg.max_k || diskann_cfg.radius_low_bound < diskann_cfg.radius_high_bound) {
            LOG_KNOWHERE_ERROR_ << "min_k should be smaller than max_k, radius_low_bound should be larger than the "
                                   "radius_high_bound in metric IP";
            return Status::invalid_args;
        }
    } else {
        return Status::invalid_metric_type;
    }
    return Status::success;
}
}  // namespace

template <typename T>
Status
DiskANNIndexNode<T>::Add(const DataSet& dataset, const Config& cfg) {
    assert(file_manager_ != nullptr);
    std::lock_guard<std::mutex> lock(preparation_lock_);
    auto build_conf = static_cast<const DiskANNConfig&>(cfg);
    CheckAddParams(build_conf);
    if (!LoadFile(build_conf.data_path)) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building." << std::endl;
        return Status::diskann_file_error;
    }
    auto& data_path = build_conf.data_path;
    index_prefix_ = build_conf.index_prefix;

    size_t count;
    size_t dim;
    diskann::get_bin_metadata(build_conf.data_path, count, dim);
    count_.store(count);
    dim_.store(dim);

    auto diskann_metric =
        build_conf.metric_type == knowhere::metric::L2 ? diskann::Metric::L2 : diskann::Metric::INNER_PRODUCT;
    diskann::BuildConfig diskann_internal_build_config{data_path,
                                                       index_prefix_,
                                                       diskann_metric,
                                                       static_cast<unsigned>(build_conf.max_degree),
                                                       static_cast<unsigned>(build_conf.search_list_size),
                                                       static_cast<double>(build_conf.pq_code_budget_gb),
                                                       static_cast<double>(build_conf.build_dram_budget_gb),
                                                       static_cast<uint32_t>(build_conf.num_threads),
                                                       static_cast<uint32_t>(build_conf.disk_pq_dims),
                                                       false,
                                                       build_conf.accelerate_build};
    auto build_stat =
        TryDiskANNCall<int>([&]() -> int { return diskann::build_disk_index<T>(diskann_internal_build_config); });

    if (!build_stat.has_value()) {
        return build_stat.error();
    } else if (build_stat.value() != 0) {
        return Status::diskann_inner_error;
    }

    // Add file to the file manager
    for (auto& filename : GetNecessaryFilenames(index_prefix_, diskann_metric == diskann::INNER_PRODUCT, true, true)) {
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
bool
DiskANNIndexNode<T>::Prepare(const Config& cfg) {
    std::lock_guard<std::mutex> lock(preparation_lock_);
    auto prep_conf = static_cast<const DiskANNConfig&>(cfg);
    if (is_prepared_.load()) {
        return true;
    }
    index_prefix_ = prep_conf.index_prefix;
    auto diskann_metric =
        prep_conf.metric_type == knowhere::metric::L2 ? diskann::Metric::L2 : diskann::Metric::INNER_PRODUCT;

    // Load file from file manager.
    for (auto& filename :
         GetNecessaryFilenames(index_prefix_, diskann_metric == diskann::INNER_PRODUCT,
                               prep_conf.search_cache_budget_gb > 0 && !prep_conf.use_bfs_cache, prep_conf.warm_up)) {
        if (!LoadFile(filename)) {
            return false;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        auto is_exist_op = file_manager_->IsExisted(filename);
        if (!is_exist_op.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
            return false;
        }
        if (is_exist_op.value() && !LoadFile(filename)) {
            return false;
        }
    }

    // set thread pool
    auto num_thread_max_value = kLinuxAioMaxnrLimit / prep_conf.aio_maxnr;
    pool_ = ThreadPool::GetGlobalThreadPool();

    // The number of threads used for preparing and searching. Threads run in parallel and one thread handles one query
    // at a time. More threads will result in higher aggregate query throughput, but will also use more IOs/second
    // across the system, which may lead to higher per-query latency. So find the balance depending on the maximum
    // number of IOPs supported by the SSD.
    if (num_thread_max_value < pool_->size()) {
        LOG_KNOWHERE_ERROR_ << "The global thread pool is too large for DiskANN. Expected max: " << num_thread_max_value
                            << ", actual:" << pool_->size();
        return false;
    }

    // load diskann pq code and meta info
    std::shared_ptr<AlignedFileReader> reader = nullptr;

    reader.reset(new LinuxAlignedFileReader(prep_conf.aio_maxnr));

    pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<T>>(reader, diskann_metric);

    auto load_expect = TryDiskANNCall<int>(
        [&]() -> int { return pq_flash_index_->load(prep_conf.num_threads, index_prefix_.c_str()); });

    if (!load_expect.has_value() || load_expect.value() != 0) {
        LOG_KNOWHERE_ERROR_ << "Failed to load DiskANN.";
        return false;
    }

    count_.store(pq_flash_index_->get_num_points());
    // DiskANN will add one more dim for IP type.
    if (diskann_metric == diskann::Metric::INNER_PRODUCT) {
        dim_.store(pq_flash_index_->get_data_dim() - 1);
    } else {
        dim_.store(pq_flash_index_->get_data_dim());
    }

    std::string warmup_query_file = diskann::get_sample_data_filename(index_prefix_);
    // load cache
    auto num_nodes_to_cache = GetCachedNodeNum(prep_conf.search_cache_budget_gb, pq_flash_index_->get_data_dim(),
                                               pq_flash_index_->get_max_degree());
    if (num_nodes_to_cache > pq_flash_index_->get_num_points() / 3) {
        LOG_KNOWHERE_ERROR_
            << "Failed to generate cache, num_nodes_to_cache is larger than 1/3 of the total data number.";
        return false;
    }
    if (num_nodes_to_cache > 0) {
        std::vector<uint32_t> node_list;
        LOG_KNOWHERE_INFO_ << "Caching " << num_nodes_to_cache << " sample nodes around medoid(s).";
        if (prep_conf.use_bfs_cache) {
            auto gen_cache_expect = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->cache_bfs_levels(num_nodes_to_cache, node_list);
                return true;
            });

            if (!gen_cache_expect.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to generate bfs cache for DiskANN.";
                return false;
            }

        } else {
            auto gen_cache_expect = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
                                                                         prep_conf.num_threads, node_list);
                return true;
            });

            if (!gen_cache_expect.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to generate cache from sample queries for DiskANN.";
                return false;
            }
        }
        auto load_cache_expect = TryDiskANNCall<bool>([&]() -> bool {
            pq_flash_index_->load_cache_list(node_list);
            return true;
        });

        if (!load_cache_expect.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to load cache for DiskANN.";
            return false;
        }
    }

    // set thread number
    omp_set_num_threads(prep_conf.num_threads);
    num_threads_ = prep_conf.num_threads;

    // warmup
    if (prep_conf.warm_up) {
        LOG_KNOWHERE_INFO_ << "Warming up.";
        uint64_t warmup_L = 20;
        uint64_t warmup_num = 0;
        uint64_t warmup_dim = 0;
        uint64_t warmup_aligned_dim = 0;
        T* warmup = nullptr;
        auto load_nodes_expect = TryDiskANNCall<bool>([&]() -> bool {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
            return true;
        });
        if (!load_nodes_expect.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for DiskANN.";
            return false;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

        bool all_searches_are_good = true;
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t)warmup_num; ++i) {
            auto search_expect = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                                    warmup_result_ids_64.data() + (i * 1),
                                                    warmup_result_dists.data() + (i * 1), 4);
                return true;
            });
            if (!search_expect.has_value()) {
                all_searches_are_good = false;
            }
        }
        if (warmup != nullptr) {
            diskann::aligned_free(warmup);
        }

        if (!all_searches_are_good) {
            LOG_KNOWHERE_ERROR_ << "Failed to do search on warmup file for DiskANN.";
            return false;
        }
    }

    is_prepared_.store(true);
    return true;
}

template <typename T>
expected<DataSetPtr, Status>
DiskANNIndexNode<T>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!is_prepared_.load()) {
        const_cast<DiskANNIndexNode<T>*>(this)->Prepare(cfg);
    }
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return unexpected(Status::empty_index);
    }

    auto search_conf = static_cast<const DiskANNConfig&>(cfg);
    auto k = static_cast<uint64_t>(search_conf.k);
    auto lsearch = static_cast<uint64_t>(search_conf.search_list_size);
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth);

    auto nq = dataset.GetRows();
    auto dim = dataset.GetDim();
    auto xq = static_cast<const T*>(dataset.GetTensor());

    feder::diskann::FederResultUniq feder_result;
    if (search_conf.trace_visit) {
        if (nq != 1) {
            return unexpected(Status::invalid_args);
        }
        feder_result = std::make_unique<feder::diskann::FederResult>();
        feder_result->visit_info_.SetQueryConfig(search_conf.k, search_conf.beamwidth, search_conf.search_list_size);
    }

    auto p_id = new int64_t[k * nq];
    auto p_dist = new float[k * nq];

    std::vector<std::future<void>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.push_back(pool_->push([&, index = row]() {
            pq_flash_index_->cached_beam_search(xq + (index * dim), k, lsearch, p_id + (index * k),
                                                p_dist + (index * k), beamwidth, false, nullptr, feder_result, bitset);
        }));
    }
    for (auto& future : futures) {
        future.get();
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
expected<DataSetPtr, Status>
DiskANNIndexNode<T>::RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!is_prepared_.load()) {
        const_cast<DiskANNIndexNode<T>*>(this)->Prepare(cfg);
    }
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return unexpected(Status::empty_index);
    }

    auto search_conf = static_cast<const DiskANNConfig&>(cfg);
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth);
    auto min_k = static_cast<uint64_t>(search_conf.min_k);
    auto max_k = static_cast<uint64_t>(search_conf.max_k);
    auto search_list_and_k_ratio = search_conf.search_list_and_k_ratio;

    auto low_bound = search_conf.radius_low_bound;
    auto high_bound = search_conf.radius_high_bound;
    bool is_ip = (pq_flash_index_->get_metric() == diskann::Metric::INNER_PRODUCT);
    float radius = (is_ip ? low_bound : high_bound);

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
    for (int64_t row = 0; row < nq; ++row) {
        futures.push_back(pool_->push([&, index = row]() {
            std::vector<int64_t> indices;
            std::vector<float> distances;
            pq_flash_index_->range_search(xq + (index * dim), radius, min_k, max_k, result_id_array[index],
                                          result_dist_array[index], beamwidth, search_list_and_k_ratio, bitset);
            // filter range search result
            FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, low_bound,
                                            high_bound);
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, low_bound, high_bound, p_dist, p_id, p_lims);
    return GenResultDataSet(nq, p_id, p_dist, p_lims);
}

template <typename T>
expected<DataSetPtr, Status>
DiskANNIndexNode<T>::GetIndexMeta(const Config& cfg) const {
    std::vector<int64_t> entry_points;
    for (size_t i = 0; i < pq_flash_index_->get_num_medoids(); i++) {
        entry_points.push_back(pq_flash_index_->get_medoids()[i]);
    }
    auto diskann_conf = static_cast<const DiskANNConfig&>(cfg);
    feder::diskann::DiskANNMeta meta(diskann_conf.data_path, diskann_conf.max_degree, diskann_conf.search_list_size,
                                     diskann_conf.pq_code_budget_gb, diskann_conf.build_dram_budget_gb,
                                     diskann_conf.num_threads, diskann_conf.disk_pq_dims, diskann_conf.accelerate_build,
                                     Count(), entry_points);
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

KNOWHERE_REGISTER_GLOBAL(DISKANNFLOAT,
                         [](const Object& object) { return Index<DiskANNIndexNode<float>>::Create(object); });
KNOWHERE_REGISTER_GLOBAL(DISKANNUINT8,
                         [](const Object& object) { return Index<DiskANNIndexNode<uint8_t>>::Create(object); });
KNOWHERE_REGISTER_GLOBAL(DISKANNINT8,
                         [](const Object& object) { return Index<DiskANNIndexNode<int8_t>>::Create(object); });
}  // namespace knowhere
