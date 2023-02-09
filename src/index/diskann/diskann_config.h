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

#ifndef DISKANN_CONFIG_H
#define DISKANN_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class DiskANNConfig : public BaseConfig {
 public:
    // Path prefix to load or save DiskANN
    CFG_STRING index_prefix;
    // The path to the raw data file. Raw data's format should be [row_num(4 bytes) | dim_num(4 bytes) | vectors].
    CFG_STRING data_path;
    // This is the degree of the graph index, typically between 60 and 150. Larger R will result in larger indices and
    // longer indexing times, but better search quality.
    CFG_INT max_degree;
    // The size of the search list during the index build or (knn/ange) search. Typical values are between 75 to 200.
    // Larger values will take more time to build but result in indices that provide higher recall for the same search
    // complexity. Plz set this value larger than the max_degree unless you need to build indices really quickly and can
    // somewhat compromise on quality.
    CFG_INT search_list_size;
    // Limit the size of the PQ code after the raw vector has been PQ-encoded. PQ code is (a search_list_size / row_num
    // )-dimensional uint8 vector. If pq_code_budget_gb is too large, it will be adjusted to the size of dim*row_num.
    CFG_FLOAT pq_code_budget_gb;
    // Limit on the memory allowed for building the index in GB. If you specify a value less than what is required to
    // build the index in one pass, the index is built using a divide and conquer approach so that sub-graphs will fit
    // in the RAM budget. The sub-graphs are overlayed to build the overall index. This approach can be up to 1.5 times
    // slower than building the index in one shot. Allocate as much memory as your RAM allows.
    CFG_FLOAT build_dram_budget_gb;
    // Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100% recall. If your vectors are
    // too large to store in SSD, this parameter provides the option to compress the vectors using PQ for storing on
    // SSD. This will trade off the recall. You would also want this to be greater than the number of bytes used for the
    // PQ compressed data stored in-memory
    CFG_INT disk_pq_dims;
    // This is the flag to enable fast build, in which we will not build vamana graph by full 2 round. This can
    // accelerate index build ~30% with an ~1% recall regression.
    CFG_BOOL accelerate_build;
    // The number of threads used for preparing and searching. When 'num_threads' uses as build parameter, the indexing
    // time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM
    // bandwidth). When 'num_threads' uses as prepare parameter, Threads run in parallel and one thread handles one
    // query at a time. More threads will result in higher aggregate query throughput, but will also use more IOs/second
    // across the system, which may lead to higher per-query latency. So find the balance depending on the maximum
    // number of IOPs supported by the SSD.
    CFG_INT num_threads;
    // While serving the index, the entire graph is stored on SSD. For faster search performance, you can cache a few
    // frequently accessed nodes in memory.
    CFG_FLOAT search_cache_budget_gb;
    // Should we do warm-up before searching.
    CFG_BOOL warm_up;
    // Should we use the bfs strategy to cache. We have two cache strategies: 1. use sample queries to do searches and
    // cached the nodes on the search paths; 2. do bfs from the entry point and cache them. The first method is suitable
    // for TopK query heavy circumstances and the second one performed better in range search.
    CFG_BOOL use_bfs_cache;
    // @deprecated
    CFG_INT aio_maxnr;
    // The beamwidth to be used for search. This is the maximum number of IO requests each query will issue per
    // iteration of search code. Larger beamwidth will result in fewer IO round-trips per query but might result in
    // slightly higher total number of IO requests to SSD per query. For the highest query throughput with a fixed SSD
    // IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
    CFG_INT beamwidth;
    // DiskANN uses TopK search to simulate range search by double the K in every round. This is the start K.
    CFG_INT min_k;
    // DiskANN uses TopK search to simulate range search by double the K in every round. This is the largest K.
    CFG_INT max_k;
    // DiskANN uses TopK search to simulate range search, this is the ratio of search list size and k. With larger
    // ratio, the accuracy will get higher but throughput will get affected.
    CFG_FLOAT search_list_and_k_ratio;
    KNOHWERE_DECLARE_CONFIG(DiskANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(index_prefix)
            .description("path to load or save Diskann.")
            .for_train()
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(data_path).description("raw data path.").for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_degree)
            .description("the degree of the graph index.")
            .set_default(48)
            .set_range(1, 2048)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_list_size)
            .description("the size of search list during the index build.")
            .set_default(128)
            .set_range(1, 65536)
            .for_train()
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb)
            .description("the size of PQ compressed representation in GB.")
            .set_range(0, std::numeric_limits<CFG_FLOAT>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(build_dram_budget_gb)
            .description("limit on the memory allowed for building the index in GB.")
            .set_range(0, std::numeric_limits<CFG_FLOAT>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(num_threads)
            .description("number of threads used by the index build/search process.")
            .set_default(8)
            .set_range(1, 256)
            .for_train()
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(disk_pq_dims)
            .description("the dimension of compressed vectors stored on the ssd, use 0 to store uncompressed data.")
            .set_default(0)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(accelerate_build)
            .description("a flag to enbale fast build.")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_cache_budget_gb)
            .description("the size of cached nodes in GB.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT>::max())
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(warm_up)
            .description("should do warm up before search.")
            .set_default(false)
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(use_bfs_cache)
            .description("should bfs strategy to cache nodes.")
            .set_default(false)
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(aio_maxnr)
            .description("the numebr of maximum parallel disk reads per thread.")
            .set_default(32)
            .set_range(1, 256)
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(beamwidth)
            .description("the maximum number of IO requests each query will issue per iteration of search code.")
            .set_default(8)
            .set_range(1, 128)
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(min_k)
            .description("the min l_search size used in range search.")
            .set_default(100)
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_k)
            .description("the max l_search size used in range search.")
            .set_default(10000)
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_list_and_k_ratio)
            .description("the ratio of search list size and k.")
            .set_default(2.0)
            .set_range(1.0, 5.0)
            .for_range_search();
    }
};
}  // namespace knowhere
#endif /* DISKANN_CONFIG_H */
