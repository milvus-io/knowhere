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

#pragma once

#include <string>

#include "knowhere/common/Config.h"

namespace knowhere {

struct DiskANNBuildConfig {
    // The path to the raw data file. Raw data's format should be [row_num(4 bytes) | dim_num(4 bytes) | vectors].
    std::string data_path;
    // This is the degree of the graph index, typically between 60 and 150. Larger R will result in larger indices and
    // longer indexing times, but better search quality.
    uint32_t max_degree;
    // The size of the search list during the index build. Typical values are between 75 to 200. Larger values will take
    // more time to build but result in indices that provide higher recall for the same search complexity. Plz set this
    // value larger than the max_degree unless you need to build indices really quickly and can somewhat compromise on
    // quality.
    uint32_t search_list_size;
    // Bound on the memory footprint of the index at search time in GB. Once built, the index will use up only the
    // specified RAM limit, the rest will reside on disk. This will dictate how aggressively we compress the data
    // vectors to store in memory. Larger will yield better performance at search time. For an n point index, to use b
    // byte PQ compressed representation in memory, use search_dram_budget_gb = ((n * b) / 2^30 + (250000 * (4 *
    // max_degree + sizeof(T)*ndim)) / 2^30). The second term in the summation is to allow some buffer for caching about
    // 250,000 nodes from the graph in memory while serving. If you are not sure about this term, add 0.25GB to the
    // first term.
    float search_dram_budget_gb;
    // Limit on the memory allowed for building the index in GB. If you specify a value less than what is required to
    // build the index in one pass, the index is built using a divide and conquer approach so that sub-graphs will fit
    // in the RAM budget. The sub-graphs are overlayed to build the overall index. This approach can be up to 1.5 times
    // slower than building the index in one shot. Allocate as much memory as your RAM allows.
    float build_dram_budget_gb;
    // Number of threads used by the index build process. Since the code is highly parallel, the indexing time improves
    // almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
    uint32_t num_threads;
    // Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100% recall. If your vectors are
    // too large to store in SSD, this parameter provides the option to compress the vectors using PQ for storing on
    // SSD. This will trade off the recall. You would also want this to be greater than the number of bytes used for the
    // PQ compressed data stored in-memory
    uint32_t pq_disk_bytes;

    static DiskANNBuildConfig
    Get(const Config& config);

    static void
    Set(Config& config, const DiskANNBuildConfig& build_conf);
};

struct DiskANNPrepareConfig {
    // The number of threads used for preparing and searching. Threads run in parallel and one thread handles one query
    // at a time. More threads will result in higher aggregate query throughput, but will also use more IOs/second
    // across the system, which may lead to higher per-query latency. So find the balance depending on the maximum
    // number of IOPs supported by the SSD.
    uint32_t num_threads;
    // While serving the index, the entire graph is stored on SSD. For faster search performance, you can cache a few
    // frequently accessed nodes in memory.
    uint32_t num_nodes_to_cache;
    // Should we do warm-up before searching.
    bool warm_up;
    // Should we use the bfs strategy to cache. We have two cache strategies: 1. use sample queries to do searches and
    // cached the nodes on the search paths; 2. do bfs from the entry point and cache them. The first method is suitable
    // for TopK query heavy circumstances and the second one performed better in range search.
    bool use_bfs_cache;

    static DiskANNPrepareConfig
    Get(const Config& config);

    static void
    Set(Config& config, const DiskANNPrepareConfig& prep_conf);
};

struct DiskANNQueryConfig {
    uint64_t k;
    // A list of search_list sizes to perform searches with. Larger parameters will result in slower latencies, but
    // higher accuracies. Must be at least the value of k.
    uint32_t search_list_size;
    // The beamwidth to be used for search. This is the maximum number of IO requests each query will issue per
    // iteration of search code. Larger beamwidth will result in fewer IO round-trips per query but might result in
    // slightly higher total number of IO requests to SSD per query. For the highest query throughput with a fixed SSD
    // IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
    uint32_t beamwidth;

    static DiskANNQueryConfig
    Get(const Config& config);

    static void
    Set(Config& config, const DiskANNQueryConfig& query_conf);
};

struct DiskANNQueryByRangeConfig {
    float radius;
    // DiskANN uses TopK search to simulate range search by double the K in every round. This is the start K.
    uint64_t min_k;
    // DiskANN uses TopK search to simulate range search by double the K in every round. This is the largest K.
    uint64_t max_k;
    // Beamwidth used for TopK search.
    uint32_t beamwidth;

    static DiskANNQueryByRangeConfig
    Get(const Config& config);

    static void
    Set(Config& config, const DiskANNQueryByRangeConfig& query_conf);
};

}  // namespace knowhere
