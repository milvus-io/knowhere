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

#include <algorithm>
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
static constexpr const char* kPQCodeBudgetGb = "pq_code_budget_gb";
static constexpr const char* kBuildDramBudgetGb = "build_dram_budget_gb";
static constexpr const char* kNumThreads = "num_threads";
static constexpr const char* kDiskPqBytes = "disk_pq_dims";
static constexpr const char* kAccelerateBuild = "accelerate_build";

static constexpr const char* kCacheDramBudgetGb = "search_cache_budget_gb";
static constexpr const char* kWarmUp = "warm_up";
static constexpr const char* kUseBfsCache = "use_bfs_cache";

static constexpr const char* kK = "k";
static constexpr const char* kBeamwidth = "beamwidth";

static constexpr const char* kRadius = "radius";
static constexpr const char* kMinK = "min_k";
static constexpr const char* kMaxK = "max_k";
static constexpr const char* kSearchListAndKRatio = "search_list_and_k_ratio";

static constexpr const char* kDiskANNBuildConfig = "diskANN_build_config";
static constexpr const char* kDiskANNPrepareConfig = "diskANN_prepare_config";
static constexpr const char* kDiskANNQueryConfig = "diskANN_query_config";
static constexpr const char* kDiskANNQueryByRangeConfig = "diskANN_query_by_range_config";

static constexpr uint32_t kMaxDegreeMinValue = 1;
static constexpr uint32_t kMaxDegreeMaxValue = 512;
static constexpr uint32_t kBuildSearchListSizeMinValue = 1;
static constexpr std::optional<uint32_t> kBuildSearchListSizeMaxValue = std::nullopt;
static constexpr float kPQCodeBudgetGbMinValue = 0;
static constexpr std::optional<float> kPQCodeBudgetGbMaxValue = std::nullopt;
static constexpr float kBuildDramBudgetGbMinValue = 0;
static constexpr std::optional<float> kBuildDramBudgetGbMaxValue = std::nullopt;
static constexpr uint32_t kBuildNumThreadsMinValue = 1;
static constexpr uint32_t kBuildNumThreadsMaxValue = 128;
static constexpr uint32_t kDiskPqBytesMinValue = 0;
static constexpr std::optional<uint32_t> kDiskPqBytesMaxValue = std::nullopt;
static constexpr uint32_t kSearchListSizeMaxValue = 200;
static constexpr uint32_t kBeamwidthMinValue = 1;
static constexpr uint32_t kBeamwidthMaxValue = 128;
static constexpr uint64_t kKMinValue = 1;
static constexpr std::optional<uint64_t> kKMaxValue = std::nullopt;
static constexpr uint32_t kSearchNumThreadsMinValue = 1;
static constexpr float kCacheDramBudgetGbMinValue = 0;
static constexpr std::optional<float> kCacheDramBudgetGbMaxValue = std::nullopt;
static constexpr std::optional<float> kRadiusMinValue = std::nullopt;
static constexpr std::optional<float> kRadiusMaxValue = std::nullopt;
static constexpr uint64_t kMinKMinValue = 1;
static constexpr std::optional<uint64_t> kMinKMaxValue = std::nullopt;
static constexpr std::optional<uint64_t> kMaxKMaxValue = std::nullopt;
static constexpr float kSearchListAndKRatioMinValue = 1.0;
static constexpr float kSearchListAndKRatioMaxValue = 5.0;
static constexpr float kSearchListAndKRatioDefaultValue = 2.0;

template <typename T>
void
CheckType(const Config& config, const std::string& key) {
    if constexpr (std::is_same_v<T, bool>) {
        if (!config[key].is_boolean()) {
            KNOWHERE_THROW_FORMAT("Param '%s' should be a bool.", key.data());
        }
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!config[key].is_string()) {
            KNOWHERE_THROW_FORMAT("Param '%s' should be a string.", key.data());
        }
    } else if constexpr (std::is_integral_v<T>) {
        if (!config[key].is_number_integer()) {
            KNOWHERE_THROW_FORMAT("Param '%s' should be a integer.", key.data());
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        if (!config[key].is_number_float()) {
            KNOWHERE_THROW_FORMAT("Param '%s' should be a float.", key.data());
        }
    } else {
        KNOWHERE_THROW_MSG("Unsupported type.");
    }
}

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
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                  "CheckNumericParamAndSet only accept int and float type");

    if (!config.contains(key)) {
        KNOWHERE_THROW_FORMAT("Param '%s' not exist", key.data());
    }

    T min = min_o.has_value() ? min_o.value() : std::numeric_limits<T>::lowest();
    T max = max_o.has_value() ? max_o.value() : std::numeric_limits<T>::max();

    CheckType<T>(config, key);

    T value = GetValueFromConfig<T>(config, key);
    if (value < min || value > max) {
        std::stringstream error_msg;
        error_msg << "Param '" << key << "'(" << value << ") is not in range [" << min << ", " << max << "]";
        KNOWHERE_THROW_MSG(error_msg.str());
    }
    config.at(key).get_to(to_be_set);
}

/**
 * @brief Check the non-numeric param's existence and type, and allocate it to the config.
 */
template <typename T>
void
CheckNonNumbericParamAndSet(const Config& config, const std::string& key, T& to_be_set) {
    if (!config.contains(key)) {
        KNOWHERE_THROW_FORMAT("Param '%s' not exist", key.data());
    }
    CheckType<T>(config, key);
    config.at(key).get_to(to_be_set);
}
}  // namespace

void
to_json(Config& config, const DiskANNBuildConfig& build_conf) {
    config = Config{{kDataPath, build_conf.data_path},
                    {kMaxDegree, build_conf.max_degree},
                    {kSearchListSize, build_conf.search_list_size},
                    {kPQCodeBudgetGb, build_conf.pq_code_budget_gb},
                    {kBuildDramBudgetGb, build_conf.build_dram_budget_gb},
                    {kNumThreads, build_conf.num_threads},
                    {kDiskPqBytes, build_conf.disk_pq_dims},
                    {kAccelerateBuild, build_conf.accelerate_build}};
}

void
from_json(const Config& config, DiskANNBuildConfig& build_conf) {
    CheckNonNumbericParamAndSet<std::string>(config, kDataPath, build_conf.data_path);
    CheckNumericParamAndSet<uint32_t>(config, kMaxDegree, kMaxDegreeMinValue, kMaxDegreeMaxValue,
                                      build_conf.max_degree);
    CheckNumericParamAndSet<uint32_t>(config, kSearchListSize, kBuildSearchListSizeMinValue,
                                      kBuildSearchListSizeMaxValue, build_conf.search_list_size);
    CheckNumericParamAndSet<float>(config, kPQCodeBudgetGb, kPQCodeBudgetGbMinValue, kPQCodeBudgetGbMaxValue,
                                   build_conf.pq_code_budget_gb);
    CheckNumericParamAndSet<float>(config, kBuildDramBudgetGb, kBuildDramBudgetGbMinValue, kBuildDramBudgetGbMaxValue,
                                   build_conf.build_dram_budget_gb);
    CheckNumericParamAndSet<uint32_t>(config, kNumThreads, kBuildNumThreadsMinValue, kBuildNumThreadsMaxValue,
                                      build_conf.num_threads);
    CheckNumericParamAndSet<uint32_t>(config, kDiskPqBytes, kDiskPqBytesMinValue, kDiskPqBytesMaxValue,
                                      build_conf.disk_pq_dims);
    CheckNonNumbericParamAndSet<bool>(config, kAccelerateBuild, build_conf.accelerate_build);
}

void
to_json(Config& config, const DiskANNPrepareConfig& prep_conf) {
    config = Config{{kNumThreads, prep_conf.num_threads},
                    {kCacheDramBudgetGb, prep_conf.search_cache_budget_gb},
                    {kWarmUp, prep_conf.warm_up},
                    {kUseBfsCache, prep_conf.use_bfs_cache}};
}

void
from_json(const Config& config, DiskANNPrepareConfig& prep_conf) {
    CheckNumericParamAndSet<uint32_t>(config, kNumThreads, kSearchNumThreadsMinValue, 2048, prep_conf.num_threads);
    CheckNumericParamAndSet<float>(config, kCacheDramBudgetGb, kCacheDramBudgetGbMinValue, kCacheDramBudgetGbMaxValue,
                                   prep_conf.search_cache_budget_gb);
    CheckNonNumbericParamAndSet<bool>(config, kWarmUp, prep_conf.warm_up);
    CheckNonNumbericParamAndSet<bool>(config, kUseBfsCache, prep_conf.use_bfs_cache);
}

void
to_json(Config& config, const DiskANNQueryConfig& query_conf) {
    config =
        Config{{kK, query_conf.k}, {kSearchListSize, query_conf.search_list_size}, {kBeamwidth, query_conf.beamwidth}};
}

void
from_json(const Config& config, DiskANNQueryConfig& query_conf) {
    CheckNumericParamAndSet<uint64_t>(config, kK, kKMinValue, kKMaxValue, query_conf.k);
    // The search_list_size should be no less than the k.
    CheckNumericParamAndSet<uint32_t>(config, kSearchListSize, query_conf.k,
                                      std::max(kSearchListSizeMaxValue, static_cast<uint32_t>(10 * query_conf.k)),
                                      query_conf.search_list_size);
    CheckNumericParamAndSet<uint32_t>(config, kBeamwidth, kBeamwidthMinValue, kBeamwidthMaxValue, query_conf.beamwidth);
}

void
to_json(Config& config, const DiskANNQueryByRangeConfig& query_conf) {
    config = Config{{kRadius, query_conf.radius},
                    {kMinK, query_conf.min_k},
                    {kMaxK, query_conf.max_k},
                    {kBeamwidth, query_conf.beamwidth},
                    {kSearchListAndKRatio, query_conf.search_list_and_k_ratio}};
}

void
from_json(const Config& config, DiskANNQueryByRangeConfig& query_conf) {
    CheckNumericParamAndSet<float>(config, kRadius, kRadiusMinValue, kRadiusMaxValue, query_conf.radius);
    CheckNumericParamAndSet<uint64_t>(config, kMinK, kMinKMinValue, kMinKMaxValue, query_conf.min_k);
    CheckNumericParamAndSet<uint64_t>(config, kMaxK, query_conf.min_k, kMaxKMaxValue, query_conf.max_k);
    CheckNumericParamAndSet<uint32_t>(config, kBeamwidth, kBeamwidthMinValue, kBeamwidthMaxValue, query_conf.beamwidth);
    if (config.contains(kSearchListAndKRatio)) {
        CheckNumericParamAndSet<float>(config, kSearchListAndKRatio, kSearchListAndKRatioMinValue,
                                       kSearchListAndKRatioMaxValue, query_conf.search_list_and_k_ratio);
    } else {
        query_conf.search_list_and_k_ratio = kSearchListAndKRatioDefaultValue;
    }
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
