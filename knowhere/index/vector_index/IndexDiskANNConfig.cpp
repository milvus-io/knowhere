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

#include "knowhere/index/vector_index/IndexDiskANNConfig.h"

#include <limits>
#include <optional>
#include <sstream>

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

namespace knowhere {
namespace {
static constexpr const char* kDataPath = "data_path";
static constexpr const char* kMaxDegree = "max_degree";
static constexpr const char* kSearchListSize = "search_list_size";
static constexpr const char* kSearchDramBudgetGb = "search_dram_budget_gb";
static constexpr const char* kBuildDramBudgetGb = "build_dram_budget_gb";
static constexpr const char* kNumThreads = "num_threads";
static constexpr const char* kPqDiskBytes = "pq_disk_bytes";

static constexpr const char* kNumNodesToCache = "num_nodes_to_cache";
static constexpr const char* kWarmUp = "warm_up";
static constexpr const char* kUseBfsCache = "use_bfs_cache";

static constexpr const char* kK = "k";
static constexpr const char* kBeamwidth = "beamwidth";

static constexpr const char* kRadius = "radius";
static constexpr const char* kMinK = "min_k";
static constexpr const char* kMaxK = "max_k";

static constexpr const char* kDiskANNBuildConfig = "diskANN_build_config";
static constexpr const char* kDiskANNPrepareConfig = "diskANN_prepare_config";
static constexpr const char* kDiskANNQueryConfig = "diskANN_query_config";
static constexpr const char* kDiskANNQueryByRangeConfig = "diskANN_query_by_range_config";

/**
 * @brief Check the numeric param's existence and type, and allocate it to the config.
 *
 * @tparam T can only be float or int.
 * @param min_o the min value of the value range, should be std::nullopt if we want to use numeric_limits::min().
 * @param max_o the max value of the value range, should be std::nullopt if we want to use numeric_limits::max().
 * @throw KnowhereException if any error.
 */
template <typename T>
void
CheckNumericParamAndSet(const Config& config, const std::string& key, std::optional<T> min_o, std::optional<T> max_o,
                        T& to_be_set) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "CheckAndSet only accept int and float type");

    if (!config.contains(key)) {
        KNOWHERE_THROW_FORMAT("Param '%s' not exist", key.data());
    }

    T min = min_o.has_value() ? min_o.value() : std::numeric_limits<T>::lowest();
    T max = max_o.has_value() ? max_o.value() : std::numeric_limits<T>::max();

    if (std::is_same_v<T, float>) {
        if (!config[key].is_number_float()) {
            KNOWHERE_THROW_FORMAT("Param '%s' should be a float", key.data());
        }
    } else {
        if (!config[key].is_number_integer()) {
            KNOWHERE_THROW_FORMAT("Param '%s' should be an integer", key.data());
        }
    }
    T value = GetValueFromConfig<T>(config, key);
    if (value < min || value > max) {
        std::stringstream error_msg;
        error_msg << "Param '" << key << "'(" << value << ") is not in range [" << min << ", " << max << "]";
        KNOWHERE_THROW_MSG(error_msg.str());
    }
    config.at(key).get_to(to_be_set);
}

/**
 * @brief Check the bool param's existence and type, and allocate it to the config.
 */
void
CheckBoolParamAndSet(const Config& config, const std::string& key, bool& to_be_set) {
    if (!config.contains(key)) {
        KNOWHERE_THROW_FORMAT("Param '%s' not exist", key.data());
    }
    if (!config[key].is_boolean()) {
        KNOWHERE_THROW_FORMAT("Param '%s' should be a bool", key.data());
    }
    config.at(key).get_to(to_be_set);
}

/**
 * @brief Check the string param's existence and type, and allocate it to the config.
 */
void
CheckStringParamAndSet(const Config& config, const std::string& key, std::string& to_be_set) {
    if (!config.contains(key)) {
        KNOWHERE_THROW_FORMAT("Param '%s' not exist", key.data());
    }
    if (!config[key].is_string()) {
        KNOWHERE_THROW_FORMAT("Param '%s' should be a string", key.data());
    }
    config.at(key).get_to(to_be_set);
}
}  // namespace

void
to_json(Config& config, const DiskANNBuildConfig& build_conf) {
    config = Config{{kDataPath, build_conf.data_path},
                    {kMaxDegree, build_conf.max_degree},
                    {kSearchListSize, build_conf.search_list_size},
                    {kSearchDramBudgetGb, build_conf.search_dram_budget_gb},
                    {kBuildDramBudgetGb, build_conf.build_dram_budget_gb},
                    {kNumThreads, build_conf.num_threads},
                    {kPqDiskBytes, build_conf.pq_disk_bytes}};
}

void
from_json(const Config& config, DiskANNBuildConfig& build_conf) {
    CheckStringParamAndSet(config, kDataPath, build_conf.data_path);
    CheckNumericParamAndSet<uint32_t>(config, kMaxDegree, 1, 512, build_conf.max_degree);
    CheckNumericParamAndSet<uint32_t>(config, kSearchListSize, 1, std::nullopt, build_conf.search_list_size);
    CheckNumericParamAndSet<float>(config, kSearchDramBudgetGb, 0, std::nullopt, build_conf.search_dram_budget_gb);
    CheckNumericParamAndSet<float>(config, kBuildDramBudgetGb, 0, std::nullopt, build_conf.build_dram_budget_gb);
    CheckNumericParamAndSet<uint32_t>(config, kNumThreads, 1, 128, build_conf.num_threads);
    CheckNumericParamAndSet<uint32_t>(config, kPqDiskBytes, 0, std::nullopt, build_conf.pq_disk_bytes);
}

void
to_json(Config& config, const DiskANNPrepareConfig& prep_conf) {
    config = Config{{kNumThreads, prep_conf.num_threads},
                    {kNumNodesToCache, prep_conf.num_nodes_to_cache},
                    {kWarmUp, prep_conf.warm_up},
                    {kUseBfsCache, prep_conf.use_bfs_cache}};
}

void
from_json(const Config& config, DiskANNPrepareConfig& prep_conf) {
    CheckNumericParamAndSet<uint32_t>(config, kNumThreads, 1, 128, prep_conf.num_threads);
    CheckNumericParamAndSet<uint32_t>(config, kNumNodesToCache, 0, std::nullopt, prep_conf.num_nodes_to_cache);
    CheckBoolParamAndSet(config, kWarmUp, prep_conf.warm_up);
    CheckBoolParamAndSet(config, kUseBfsCache, prep_conf.use_bfs_cache);
}

void
to_json(Config& config, const DiskANNQueryConfig& query_conf) {
    config =
        Config{{kK, query_conf.k}, {kSearchListSize, query_conf.search_list_size}, {kBeamwidth, query_conf.beamwidth}};
}

void
from_json(const Config& config, DiskANNQueryConfig& query_conf) {
    CheckNumericParamAndSet<uint64_t>(config, kK, 1, std::nullopt, query_conf.k);
    // The search_list_size should be no less than the k.
    CheckNumericParamAndSet<uint32_t>(config, kSearchListSize, query_conf.k, std::nullopt, query_conf.search_list_size);
    CheckNumericParamAndSet<uint32_t>(config, kBeamwidth, 1, 128, query_conf.beamwidth);
}

void
to_json(Config& config, const DiskANNQueryByRangeConfig& query_conf) {
    config = Config{{kRadius, query_conf.radius},
                    {kMinK, query_conf.min_k},
                    {kMaxK, query_conf.max_k},
                    {kBeamwidth, query_conf.beamwidth}};
}

void
from_json(const Config& config, DiskANNQueryByRangeConfig& query_conf) {
    CheckNumericParamAndSet<float>(config, kRadius, std::nullopt, std::nullopt, query_conf.radius);
    CheckNumericParamAndSet<uint64_t>(config, kMinK, 1, std::nullopt, query_conf.min_k);
    CheckNumericParamAndSet<uint64_t>(config, kMaxK, query_conf.min_k, std::nullopt, query_conf.max_k);
    CheckNumericParamAndSet<uint32_t>(config, kBeamwidth, 1, 128, query_conf.beamwidth);
}

DiskANNBuildConfig
DiskANNBuildConfig::Get(const Config& config) {
    return config.at(kDiskANNBuildConfig).get<DiskANNBuildConfig>();
}

void
DiskANNBuildConfig::Set(Config& config, const DiskANNBuildConfig& build_conf) {
    config[kDiskANNBuildConfig] = build_conf;
}

DiskANNPrepareConfig
DiskANNPrepareConfig::Get(const Config& config) {
    return config.at(kDiskANNPrepareConfig).get<DiskANNPrepareConfig>();
}

void
DiskANNPrepareConfig::Set(Config& config, const DiskANNPrepareConfig& prep_conf) {
    config[kDiskANNPrepareConfig] = prep_conf;
}

DiskANNQueryConfig
DiskANNQueryConfig::Get(const Config& config) {
    return config.at(kDiskANNQueryConfig).get<DiskANNQueryConfig>();
}

void
DiskANNQueryConfig::Set(Config& config, const DiskANNQueryConfig& query_conf) {
    config[kDiskANNQueryConfig] = query_conf;
}

DiskANNQueryByRangeConfig
DiskANNQueryByRangeConfig::Get(const Config& config) {
    return config.at(kDiskANNQueryByRangeConfig).get<DiskANNQueryByRangeConfig>();
}

void
DiskANNQueryByRangeConfig::Set(Config& config, const DiskANNQueryByRangeConfig& query_conf) {
    config[kDiskANNQueryByRangeConfig] = query_conf;
}
}  // namespace knowhere
