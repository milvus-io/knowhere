// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "index/vector_index/IndexDiskANN.h"

#include <omp.h>

#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "DiskANN/include/aux_utils.h"
#include "DiskANN/include/utils.h"
#ifndef _WINDOWS
#include "DiskANN/include/linux_aligned_file_reader.h"
#else
#include "DiskANN/include/windows_aligned_file_reader.h"
#endif
#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

namespace knowhere {

template <typename T>
IndexDiskANN<T>::IndexDiskANN(std::string index_prefix, MetricType metric_type,
                              std::unique_ptr<FileManager> file_manager)
    : index_prefix_(index_prefix), file_manager_(std::move(file_manager)) {
    index_type_ = IndexEnum::INDEX_DISKANN;

    if (metric_type == metric::L2) {
        metric_ = diskann::L2;
    } else if (metric_type == metric::IP) {
        if (!std::is_same_v<T, float>) {
            KNOWHERE_THROW_MSG(
                "DiskANN currently only supports floating point data for Max "
                "Inner Product Search. ");
        }
        metric_ = diskann::INNER_PRODUCT;
    } else {
        KNOWHERE_THROW_MSG("DiskANN only support L2 and IP distance.");
    }
}

namespace {
void
CheckPreparation(bool is_prepared) {
    if (!is_prepared) {
        KNOWHERE_THROW_MSG("DiskANN is not prepared yet, plz call Prepare() to make it ready for queries.");
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

template <typename T>
void
CheckFileSize(const std::string& data_path, const size_t num, const size_t dim) {
    std::ifstream file(data_path.c_str(), std::ios::binary | std::ios::ate);
    uint64_t autual_file_size = static_cast<uint64_t>(file.tellg());
    file.close();
    uint64_t expected_file_size = num * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (autual_file_size != expected_file_size) {
        KNOWHERE_THROW_FORMAT("Actual file size (%ld bytes) not equal to expected size (%ld bytes)", autual_file_size,
                              expected_file_size);
    }
}

/**
 * @brief Check the length of a node.
 */
template <typename T>
void
CheckNodeLength(const uint32_t degree, const size_t dim) {
    uint64_t node_length = ((degree + 1) * sizeof(unsigned) + dim * sizeof(T));
    if (node_length > SECTOR_LEN) {
        KNOWHERE_THROW_FORMAT("Node length (%ld bytes) exceeds the sector length", node_length);
    }
}

template <typename T>
void
CheckBuildParams(const DiskANNBuildConfig& build_conf) {
    size_t num, dim;
    diskann::get_bin_metadata(build_conf.data_path, num, dim);

    CheckFileSize<T>(build_conf.data_path, num, dim);
    CheckNodeLength<T>(build_conf.max_degree, dim);
}

template <typename T>
std::optional<T>
TryDiskANNCall(std::function<T()>&& diskann_call) {
    try {
        return std::make_optional<T>(diskann_call());
    } catch (const diskann::FileException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN File Exception: " << e.what();
    } catch (const diskann::ANNException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Exception: " << e.what();
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Other Exception: " << e.what();
    }
    return std::nullopt;
}

template <typename T>
T
TryDiskANNCallAndThrow(std::function<T()>&& diskann_call) {
    try {
        return diskann_call();
    } catch (const diskann::FileException& e) {
        KNOWHERE_THROW_MSG("DiskANN File Exception: " + std::string(e.what()));
    } catch (const diskann::ANNException& e) {
        KNOWHERE_THROW_MSG("DiskANN Exception: " + std::string(e.what()));
    } catch (const std::exception& e) {
        KNOWHERE_THROW_MSG("DiskANN Other Exception: " + std::string(e.what()));
    }
}
}  // namespace

template <typename T>
void
IndexDiskANN<T>::AddWithoutIds(const DatasetPtr& data_set, const Config& config) {
    auto build_conf = DiskANNBuildConfig::Get(config);

    CheckBuildParams<T>(build_conf);

    auto& data_path = build_conf.data_path;
    // Load raw data
    if (!LoadFile(data_path)) {
        KNOWHERE_THROW_MSG("Failed load the raw data before building.");
    }

    std::stringstream stream;
    stream << build_conf.max_degree << " " << build_conf.search_list_size << " " << build_conf.search_dram_budget_gb
           << " " << build_conf.build_dram_budget_gb << " " << build_conf.num_threads << " "
           << build_conf.pq_disk_bytes;

    auto build_successful = TryDiskANNCallAndThrow<int>([&]() -> int {
        return diskann::build_disk_index<T>(data_path.c_str(), index_prefix_.c_str(), stream.str().c_str(), metric_);
    });

    if (build_successful != 0) {
        KNOWHERE_THROW_MSG("Failed to build DiskANN.");
    }

    // Add file to the file manager
    for (auto& filename : GetNecessaryFilenames(index_prefix_, metric_ == diskann::INNER_PRODUCT, true, true)) {
        if (!AddFile(filename)) {
            KNOWHERE_THROW_MSG("Failed to add file " + filename + ".");
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        if (file_exists(filename) && !AddFile(filename)) {
            KNOWHERE_THROW_MSG("Failed to add file " + filename + ".");
        }
    }
}

template <typename T>
bool
IndexDiskANN<T>::Prepare(const Config& config) {
    std::lock_guard<std::mutex> lock(preparation_lock_);

    auto prep_conf = DiskANNPrepareConfig::Get(config);
    if (is_prepared_) {
        return true;
    }

    // Load file from file manager.
    for (auto& filename :
         GetNecessaryFilenames(index_prefix_, metric_ == diskann::INNER_PRODUCT,
                               prep_conf.num_nodes_to_cache > 0 && !prep_conf.use_bfs_cache, prep_conf.warm_up)) {
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

    // load PQ file
    LOG_KNOWHERE_INFO_ << "Loading PQ from disk.";
    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<T>>(reader, metric_);

    auto load_successful = TryDiskANNCall<int>(
        [&]() -> int { return pq_flash_index_->load(prep_conf.num_threads, index_prefix_.c_str()); });

    if (!load_successful.has_value() || load_successful.value() != 0) {
        LOG_KNOWHERE_ERROR_ << "Failed to load DiskANN.";
        return false;
    }

    std::string warmup_query_file = diskann::get_sample_data_filename(index_prefix_);
    // load cache
    if (prep_conf.num_nodes_to_cache > 0) {
        std::vector<uint32_t> node_list;
        LOG_KNOWHERE_INFO_ << "Caching " << prep_conf.num_nodes_to_cache << " sample nodes around medoid(s).";

        if (prep_conf.use_bfs_cache) {
            auto gen_cache_successful = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->cache_bfs_levels(prep_conf.num_nodes_to_cache, node_list);
                return true;
            });

            if (!gen_cache_successful.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to generate bfs cache for DiskANN.";
                return false;
            }
        } else {
            auto gen_cache_successful = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->generate_cache_list_from_sample_queries(
                    warmup_query_file, 15, 6, prep_conf.num_nodes_to_cache, prep_conf.num_threads, node_list);
                return true;
            });

            if (!gen_cache_successful.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to generate cache from sample queries for DiskANN.";
                return false;
            }
        }
        auto load_cache_successful = TryDiskANNCall<bool>([&]() -> bool {
            pq_flash_index_->load_cache_list(node_list);
            return true;
        });

        if (!load_cache_successful.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to load cache for DiskANN.";
            return false;
        }
    }

    // set thread number
    omp_set_num_threads(prep_conf.num_threads);

    // warmup
    if (prep_conf.warm_up) {
        LOG_KNOWHERE_INFO_ << "Warming up.";
        uint64_t warmup_L = 20;
        uint64_t warmup_num = 0;
        uint64_t warmup_dim = 0;
        uint64_t warmup_aligned_dim = 0;
        T* warmup = nullptr;
        auto load_successful = TryDiskANNCall<bool>([&]() -> bool {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
            return true;
        });
        if (!load_successful.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for DiskANN.";
            return false;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

        bool all_searches_are_good = true;
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t)warmup_num; ++i) {
            auto search_successful = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                                    warmup_result_ids_64.data() + (i * 1),
                                                    warmup_result_dists.data() + (i * 1), 4);
                return true;
            });
            if (!search_successful.has_value()) {
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

    is_prepared_ = true;
    return true;
}

template <typename T>

DatasetPtr
IndexDiskANN<T>::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    CheckPreparation(is_prepared_);

    auto query_conf = DiskANNQueryConfig::Get(config);
    auto& k = query_conf.k;

    GET_TENSOR_DATA_DIM(dataset_ptr);
    auto query = static_cast<const T*>(p_data);
    auto p_id = static_cast<int64_t*>(malloc(sizeof(int64_t) * k * rows));
    auto p_dist = static_cast<float*>(malloc(sizeof(float) * k * rows));

    std::optional<KnowhereException> ex = std::nullopt;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t row = 0; row < rows; ++row) {
        // Openmp can not throw exception out.
        try {
            TryDiskANNCallAndThrow<void>([&]() -> void {
                pq_flash_index_->cached_beam_search(query + (row * dim), k, query_conf.search_list_size,
                                                    p_id + (row * k), p_dist + (row * k), query_conf.beamwidth, false,
                                                    nullptr, bitset);
            });
        } catch (const KnowhereException& e) {
#pragma omp critical
            {
                if (ex == std::nullopt) {
                    ex = std::make_optional<KnowhereException>(e);
                }
            }
        }
    }

    if (ex.has_value()) {
        throw ex.value();
    }

    return GenResultDataset(p_id, p_dist);
}

template <typename T>
DatasetPtr
IndexDiskANN<T>::QueryByRange(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    CheckPreparation(is_prepared_);

    auto query_conf = DiskANNQueryByRangeConfig::Get(config);
    auto& radius = query_conf.radius;

    GET_TENSOR_DATA_DIM(dataset_ptr);
    auto query = static_cast<const T*>(p_data);

    std::vector<std::vector<int64_t>> result_id_array(rows);
    std::vector<std::vector<float>> result_dist_array(rows);
    auto p_lims = static_cast<size_t*>(malloc((rows + 1) * sizeof(size_t)));
    *p_lims = 0;

    std::optional<KnowhereException> ex = std::nullopt;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t row = 0; row < rows; ++row) {
        std::vector<int64_t> indices;
        std::vector<float> distances;

        auto res_count = 0;
        try {
            res_count = TryDiskANNCallAndThrow<uint32_t>([&]() -> uint32_t {
                return pq_flash_index_->range_search(query + (row * dim), radius, query_conf.min_k, query_conf.max_k,
                                                     indices, distances, query_conf.beamwidth, bitset);
            });
        } catch (const KnowhereException& e) {
#pragma omp critical
            {
                if (ex == std::nullopt) {
                    ex = std::make_optional<KnowhereException>(e);
                }
            }
        }

        result_id_array[row].resize(res_count);
        result_dist_array[row].resize(res_count);
        for (int32_t res_num = 0; res_num < res_count; ++res_num) {
            result_id_array[row][res_num] = indices[res_num];
            result_dist_array[row][res_num] = distances[res_num];
        }
        *(p_lims + row + 1) = res_count;
    }

    if (ex.has_value()) {
        throw ex.value();
    }

    for (int64_t row = 0; row < rows; ++row) {
        *(p_lims + row + 1) += *(p_lims + row);
    }

    auto ans_size = *(p_lims + rows);
    auto p_id = static_cast<int64_t*>(malloc(ans_size * sizeof(int64_t)));
    auto p_dist = static_cast<float*>(malloc(ans_size * sizeof(float)));

    for (int64_t row = 0; row < rows; ++row) {
        auto start = *(p_lims + row);
        memcpy(p_id + start, result_id_array[row].data(), result_id_array[row].size() * sizeof(int64_t));
        memcpy(p_dist + start, result_dist_array[row].data(), result_dist_array[row].size() * sizeof(float));
    }

    return GenResultDataset(p_id, p_dist, p_lims);
}

template <typename T>
int64_t
IndexDiskANN<T>::Count() {
    CheckPreparation(is_prepared_);
    return pq_flash_index_->get_num_points();
}

template <typename T>
int64_t
IndexDiskANN<T>::Dim() {
    CheckPreparation(is_prepared_);
    return pq_flash_index_->get_data_dim();
}

template <typename T>
bool
IndexDiskANN<T>::LoadFile(const std::string& filename) {
    if (!file_manager_->LoadFile(filename)) {
        LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
        return false;
    }
    return true;
}

template <typename T>
bool
IndexDiskANN<T>::AddFile(const std::string& filename) {
    if (!file_manager_->AddFile(filename)) {
        LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
        return false;
    }
    return true;
}

// Explicit template instantiation
template class IndexDiskANN<float>;
template class IndexDiskANN<uint8_t>;
template class IndexDiskANN<int8_t>;
}  // namespace knowhere
