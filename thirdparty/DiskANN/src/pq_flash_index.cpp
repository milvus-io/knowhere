// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "diskann/aligned_file_reader.h"
#include "diskann/logger.h"
#include "diskann/pq_flash_index.h"
#include <malloc.h>
#include "diskann/percentile_stats.h"

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iterator>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>
#include "diskann/distance.h"
#include "diskann/exceptions.h"
#include "diskann/parameters.h"
#include "diskann/timer.h"
#include "diskann/utils.h"
#include "diskann/defines_export.h"
#include "knowhere/heap.h"

#include "knowhere/log.h"
#include "tsl/robin_set.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "diskann/linux_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_U32(stream, val) stream.read((char *) &val, sizeof(_u32))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + disk_bytes_per_point)

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) \
  (((_u64) (id)) / nvecs_per_sector + reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) \
  ((((_u64) (id)) % nvecs_per_sector) * data_dim * sizeof(float))

namespace {
  void aggregate_coords(const unsigned *ids, const _u64 n_ids,
                        const _u8 *all_coords, const _u64 ndims, _u8 *out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts,
                      const _u64 pq_nchunks, const float *pq_dists,
                      float *dists_out) {
#if defined(__ARM_NEON__) || defined(__aarch64__)
    __builtin_prefetch((char *) dists_out, 1, 3);
    __builtin_prefetch((char *) pq_ids, 0, 3);
    __builtin_prefetch((char *) (pq_ids + 64), 0, 3);
    __builtin_prefetch((char *) (pq_ids + 128), 0, 3);
#else
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
#endif
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float *chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
#if defined(__ARM_NEON__) || defined(__aarch64__)
        __builtin_prefetch((char *) (chunk_dists + 256), 0, 3);
#else
        _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
#endif
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }
  constexpr _u64 kBruteForceBeamWidthFactor = 1; // TODO: after experiment, change this value
}  // namespace

namespace diskann {
  template<typename T>
  PQFlashIndex<T>::PQFlashIndex(std::shared_ptr<AlignedFileReader> fileReader,
                                diskann::Metric                    m)
      : reader(fileReader), metric(m) {
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = diskann::Metric::L2;
      } else {
        LOG(WARNING) << "Cannot normalize integral data types."
                     << " This may result in erroneous results or poor recall."
                     << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp = diskann::get_distance_function<T>(m);
    this->dist_cmp_float = diskann::get_distance_function<float>(m);
  }

  template<typename T>
  PQFlashIndex<T>::~PQFlashIndex() {
#ifndef EXEC_ENV_OLS
    if (data != nullptr) {
      delete[] data;
    }
#endif

    if (medoids != nullptr) {
      delete[] medoids;
    }
    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr) {
      delete[] nhood_cache_buf;
      diskann::aligned_free(coord_cache_buf);
    }

    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
    }
  }

  template<typename T>
  void PQFlashIndex<T>::setup_thread_data(_u64 nthreads) {
    LOG(INFO) << "Setting up thread-specific contexts for nthreads: "
              << nthreads;
    for (_s64 thread = 0; thread < (_s64) nthreads; thread++) {
      QueryScratch<T> scratch;
      _u64 coord_alloc_size = ROUND_UP(sizeof(T) * this->aligned_dim, 256);
      diskann::alloc_aligned((void **) &scratch.coord_scratch, coord_alloc_size,
                             256);
      diskann::alloc_aligned((void **) &scratch.sector_scratch,
                             (_u64) MAX_N_SECTOR_READS * read_len_for_node,
                             SECTOR_LEN);
      diskann::alloc_aligned(
          (void **) &scratch.aligned_pq_coord_scratch,
          (_u64) MAX_GRAPH_DEGREE * (_u64) this->aligned_dim * sizeof(_u8),
          256);
      diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                             256 * (_u64) this->aligned_dim * sizeof(float),
                             256);
      diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                             (_u64) MAX_GRAPH_DEGREE * sizeof(float), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_query_T,
                             this->aligned_dim * sizeof(T), 8 * sizeof(T));
      diskann::alloc_aligned((void **) &scratch.aligned_query_float,
                             this->aligned_dim * sizeof(float),
                             8 * sizeof(float));
      scratch.visited = new tsl::robin_set<_u64>(4096);

      memset(scratch.coord_scratch, 0, sizeof(T) * this->aligned_dim);
      memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
      memset(scratch.aligned_query_float, 0, this->aligned_dim * sizeof(float));

      ThreadData<T> data;
      data.scratch = scratch;
      this->thread_data.push(data);
    }
    load_flag = true;
  }

  template<typename T>
  void PQFlashIndex<T>::destroy_thread_data() {
    LOG_KNOWHERE_DEBUG_ << "Clearing scratch";
    assert(this->thread_data.size() == this->max_nthreads);
    while (this->thread_data.size() > 0) {
      ThreadData<T> data = this->thread_data.pop();
      while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
      }
      auto &scratch = data.scratch;
      diskann::aligned_free((void *) scratch.coord_scratch);
      diskann::aligned_free((void *) scratch.sector_scratch);
      diskann::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      diskann::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_query_float);
      diskann::aligned_free((void *) scratch.aligned_query_T);

      delete scratch.visited;
    }
  }

  template<typename T>
  void PQFlashIndex<T>::load_cache_list(std::vector<uint32_t> &node_list) {
    LOG_KNOWHERE_DEBUG_ << "Loading the cache list into memory...";
    _u64 num_cached_nodes = node_list.size();

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    auto ctx = this->reader->get_ctx();

    nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
    memset(nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

    _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
    diskann::alloc_aligned((void **) &coord_cache_buf,
                           coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++) {
      _u64 start_idx = block * BLOCK_SIZE;
      _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
      std::vector<AlignedRead>             read_reqs;
      std::vector<std::pair<_u32, char *>> nhoods;
      for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++) {
        AlignedRead read;
        char       *buf = nullptr;
        alloc_aligned((void **) &buf, read_len_for_node, SECTOR_LEN);
        nhoods.push_back(std::make_pair(node_list[node_idx], buf));
        read.len = read_len_for_node;
        read.buf = buf;
        read.offset = get_node_sector_offset(node_list[node_idx]);
        read_reqs.push_back(read);
      }

      reader->read(read_reqs, ctx);

      _u64 node_idx = start_idx;
      for (_u32 i = 0; i < read_reqs.size(); i++) {
#if defined(_WINDOWS) && \
    defined(USE_BING_INFRA)  // this block is to handle failed reads in
                             // production settings
        if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS) {
          continue;
        }
#endif
        auto &nhood = nhoods[i];
        char *node_buf = get_offset_to_node(nhood.second, nhood.first);
        T    *node_coords = OFFSET_TO_NODE_COORDS(node_buf);
        T    *cached_coords = coord_cache_buf + node_idx * aligned_dim;
        memcpy(cached_coords, node_coords, disk_bytes_per_point);
        coord_cache.insert(std::make_pair(nhood.first, cached_coords));

        // insert node nhood into nhood_cache
        unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);

        auto                        nnbrs = *node_nhood;
        unsigned                   *nbrs = node_nhood + 1;
        std::pair<_u32, unsigned *> cnhood;
        cnhood.first = nnbrs;
        cnhood.second = nhood_cache_buf + node_idx * (max_degree + 1);
        memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
        nhood_cache.insert(std::make_pair(nhood.first, cnhood));
        aligned_free(nhood.second);
        node_idx++;
      }
    }
    // return thread data
    this->thread_data.push(this_thread_data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);
    LOG_KNOWHERE_DEBUG_ << "done.";
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  void PQFlashIndex<T>::generate_cache_list_from_sample_queries(
      MemoryMappedFiles &files, std::string sample_bin, _u64 l_search,
      _u64 beamwidth, _u64 num_nodes_to_cache,
      std::vector<uint32_t> &node_list) {
#else
  template<typename T>
  void PQFlashIndex<T>::generate_cache_list_from_sample_queries(
      std::string sample_bin, _u64 l_search, _u64 beamwidth,
      _u64 num_nodes_to_cache, std::vector<uint32_t> &node_list) {
#endif
    this->count_visited_nodes = true;
    this->node_visit_counter.clear();
    this->node_visit_counter.resize(this->num_points);
    for (_u32 i = 0; i < node_visit_counter.size(); i++) {
      this->node_visit_counter[i].first = i;
      this->node_visit_counter[i].second = 0;
    }

    _u64 sample_num, sample_dim, sample_aligned_dim;
    T   *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin)) {
      diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num,
                                   sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin)) {
      diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim,
                                   sample_aligned_dim);
    }
#endif
    else {
      diskann::cerr << "Sample bin file not found. Not generating cache."
                    << std::endl;
      return;
    }

    std::vector<int64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float>   tmp_result_dists(sample_num, 0);

    auto thread_pool = knowhere::ThreadPool::GetGlobalThreadPool();
    std::vector<std::future<void>> futures;
    futures.reserve(sample_num);
    for (_s64 i = 0; i < (int64_t) sample_num; i++) {
      futures.push_back(thread_pool->push([&, index = i]() {
        cached_beam_search(samples + (index * sample_aligned_dim), 1, l_search,
                           tmp_result_ids_64.data() + (index * 1),
                           tmp_result_dists.data() + (index * 1), beamwidth);
      }));
    }

    for (auto &future : futures) {
      future.get();
    }

    std::sort(this->node_visit_counter.begin(), node_visit_counter.end(),
              [](std::pair<_u32, _u32> &left, std::pair<_u32, _u32> &right) {
                return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    node_list.reserve(num_nodes_to_cache);
    for (_u64 i = 0; i < num_nodes_to_cache; i++) {
      node_list.push_back(this->node_visit_counter[i].first);
    }
    this->count_visited_nodes = false;
    std::vector<std::pair<_u32, _u32>>().swap(this->node_visit_counter);

    diskann::aligned_free(samples);
  }

  template<typename T>
  void PQFlashIndex<T>::cache_bfs_levels(_u64 num_nodes_to_cache,
                                         std::vector<uint32_t> &node_list) {
    std::random_device rng;
    std::mt19937       urng(rng());

    node_list.clear();

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    auto ctx = this->reader->get_ctx();

    std::unique_ptr<tsl::robin_set<unsigned>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<unsigned>>();
    prev_level = std::make_unique<tsl::robin_set<unsigned>>();

    for (_u64 miter = 0; miter < num_medoids; miter++) {
      cur_level->insert(medoids[miter]);
    }

    _u64     lvl = 1;
    uint64_t prev_node_list_size = 0;
    while ((node_list.size() + cur_level->size() < num_nodes_to_cache) &&
           cur_level->size() != 0) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      std::vector<unsigned> nodes_to_expand;

      for (const unsigned &id : *prev_level) {
        if (std::find(node_list.begin(), node_list.end(), id) !=
            node_list.end()) {
          continue;
        }
        node_list.push_back(id);
        nodes_to_expand.push_back(id);
      }

      std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);

      LOG_KNOWHERE_DEBUG_ << "Level: " << lvl;
      bool finish_flag = false;

      uint64_t BLOCK_SIZE = 1024;
      uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
      for (size_t block = 0; block < nblocks && !finish_flag; block++) {
        size_t start = block * BLOCK_SIZE;
        size_t end =
            (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());
        std::vector<AlignedRead>             read_reqs;
        std::vector<std::pair<_u32, char *>> nhoods;
        for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
          char *buf = nullptr;
          alloc_aligned((void **) &buf, read_len_for_node, SECTOR_LEN);
          nhoods.push_back(std::make_pair(nodes_to_expand[cur_pt], buf));
          AlignedRead read;
          read.len = read_len_for_node;
          read.buf = buf;
          read.offset = get_node_sector_offset(nodes_to_expand[cur_pt]);
          read_reqs.push_back(read);
        }

        // issue read requests
        reader->read(read_reqs, ctx);

        // process each nhood buf
        for (_u32 i = 0; i < read_reqs.size(); i++) {
#if defined(_WINDOWS) && \
    defined(USE_BING_INFRA)  // this block is to handle read failures in
                             // production settings
          if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS) {
            continue;
          }
#endif
          auto &nhood = nhoods[i];

          // insert node coord into coord_cache
          char     *node_buf = get_offset_to_node(nhood.second, nhood.first);
          unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
          _u64      nnbrs = (_u64) *node_nhood;
          unsigned *nbrs = node_nhood + 1;
          // explore next level
          for (_u64 j = 0; j < nnbrs && !finish_flag; j++) {
            if (std::find(node_list.begin(), node_list.end(), nbrs[j]) ==
                node_list.end()) {
              cur_level->insert(nbrs[j]);
            }
            if (cur_level->size() + node_list.size() >= num_nodes_to_cache) {
              finish_flag = true;
            }
          }
          aligned_free(nhood.second);
        }
      }

      LOG_KNOWHERE_DEBUG_ << ". #nodes: "
                          << node_list.size() - prev_node_list_size
                          << ", #nodes thus far: " << node_list.size();
      prev_node_list_size = node_list.size();
      lvl++;
    }

    std::vector<uint32_t> cur_level_node_list;
    for (const unsigned &p : *cur_level)
      cur_level_node_list.push_back(p);

    std::shuffle(cur_level_node_list.begin(), cur_level_node_list.end(), urng);
    size_t residual = num_nodes_to_cache - node_list.size();

    for (size_t i = 0; i < (std::min)(residual, cur_level_node_list.size());
         i++)
      node_list.push_back(cur_level_node_list[i]);

    LOG_KNOWHERE_DEBUG_ << "Level: " << lvl << ". #nodes: "
                        << node_list.size() - prev_node_list_size
                        << ", #nodes thus far: " << node_list.size();

    // return thread data
    this->thread_data.push(this_thread_data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);

    LOG(INFO) << "done";
  }

  template<typename T>
  void PQFlashIndex<T>::use_medoids_data_as_centroids() {
    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    alloc_aligned(((void **) &centroid_data),
                  num_medoids * aligned_dim * sizeof(float), 32);
    std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    auto ctx = this->reader->get_ctx();
    // borrow buf
    auto scratch = &(data.scratch);
    scratch->reset();
    char *sector_scratch = scratch->sector_scratch;
    T    *medoid_coords = scratch->coord_scratch;

    LOG(INFO) << "Loading centroid data from medoids vector data of "
              << num_medoids << " medoid(s)";
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
      auto medoid = medoids[cur_m];
      // read medoid nhood
      std::vector<AlignedRead> medoid_read(1);
      medoid_read[0].len = read_len_for_node;
      medoid_read[0].buf = sector_scratch;
      medoid_read[0].offset = get_node_sector_offset(medoid);
      reader->read(medoid_read, ctx);

      // all data about medoid
      char *medoid_node_buf = get_offset_to_node(sector_scratch, medoid);

      // add medoid coords to `coord_cache`
      T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
      memcpy(medoid_coords, medoid_disk_coords, disk_bytes_per_point);

      if (!use_disk_index_pq) {
        for (uint32_t i = 0; i < data_dim; i++)
          centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];
      } else {
        disk_pq_table.inflate_vector((_u8 *) medoid_coords,
                                     (centroid_data + cur_m * aligned_dim));
      }
    }

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  int PQFlashIndex<T>::load(MemoryMappedFiles &files, uint32_t num_threads,
                            const char *index_prefix) {
#else
  template<typename T>
  int PQFlashIndex<T>::load(uint32_t num_threads, const char *index_prefix) {
#endif
    std::string pq_table_bin =
        get_pq_pivots_filename(std::string(index_prefix));
    std::string pq_compressed_vectors =
        get_pq_compressed_filename(std::string(index_prefix));
    std::string disk_index_file =
        get_disk_index_filename(std::string(index_prefix));
    std::string medoids_file =
        get_disk_index_medoids_filename(std::string(disk_index_file));
    std::string centroids_file =
        get_disk_index_centroids_filename(std::string(disk_index_file));

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim);
#endif

    this->disk_index_file = disk_index_file;

    if (pq_file_num_centroids != 256) {
      LOG(ERROR) << "Error. Number of PQ centroids is not 256. Exitting.";
      return -1;
    }

    this->data_dim = pq_file_dim;
    // will reset later if we use PQ on disk
    this->disk_data_dim = this->data_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->disk_bytes_per_point = this->data_dim * sizeof(T);
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<_u8>(files, pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);
#else
    diskann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);
#endif

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

#ifdef EXEC_ENV_OLS
    pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    LOG(INFO)
        << "Loaded PQ centroids and in-memory compressed vectors. #points: "
        << num_points << " #dim: " << data_dim
        << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks;

    std::string disk_pq_pivots_path = this->disk_index_file + "_pq_pivots.bin";
    if (file_exists(disk_pq_pivots_path)) {
      use_disk_index_pq = true;
#ifdef EXEC_ENV_OLS
      // giving 0 chunks to make the pq_table infer from the
      // chunk_offsets file the correct value
      disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
      // giving 0 chunks to make the pq_table infer from the
      // chunk_offsets file the correct value
      disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
      disk_pq_n_chunks = disk_pq_table.get_num_chunks();
      disk_bytes_per_point =
          disk_pq_n_chunks *
          sizeof(_u8);  // revising disk_bytes_per_point since DISK PQ is used.
      std::cout << "Disk index uses PQ data compressed down to "
                << disk_pq_n_chunks << " bytes per point." << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file reader approach.
    reader->open(disk_index_file);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    char                    *bytes = getHeaderBytes();
    ContentBuf               buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(disk_index_file, std::ios::binary);
#endif

    size_t actual_index_size = get_file_size(disk_index_file);
    size_t expected_file_size;
    READ_U64(index_metadata, expected_file_size);
    if (actual_index_size != expected_file_size) {
      LOG(ERROR) << "File size mismatch for " << disk_index_file
                 << " (size: " << actual_index_size << ")"
                 << " with meta-data size: " << expected_file_size;
      return -1;
    }

    _u64 disk_nnodes;
    READ_U64(index_metadata, disk_nnodes);
    if (disk_nnodes != num_points) {
      LOG(ERROR) << "Mismatch in #points for compressed data file and disk "
                    "index file: "
                 << disk_nnodes << " vs " << num_points;
      return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);

    if (max_node_len > SECTOR_LEN) {
      long_node = true;
      nsectors_per_node = ROUND_UP(max_node_len, SECTOR_LEN) / SECTOR_LEN;
      read_len_for_node = SECTOR_LEN * nsectors_per_node;
    }

    max_degree = ((max_node_len - disk_bytes_per_point) / sizeof(unsigned)) - 1;

    if (max_degree > MAX_GRAPH_DEGREE) {
      std::stringstream stream;
      stream << "Error loading index. Ensure that max graph degree (R) does "
                "not exceed "
             << MAX_GRAPH_DEGREE << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->num_frozen_points);
    _u64 file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->num_frozen_points == 1)
      this->frozen_location = file_frozen_id;
    if (this->num_frozen_points == 1) {
      diskann::cout << " Detected frozen point in index at location "
                    << this->frozen_location
                    << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->reorder_data_exists);
    if (this->reorder_data_exists) {
      if (this->use_disk_index_pq == false) {
        throw ANNException(
            "Reordering is designed for used with disk PQ compression option",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      READ_U64(index_metadata, this->reorder_data_start_sector);
      READ_U64(index_metadata, this->ndims_reorder_vecs);
      READ_U64(index_metadata, this->nvecs_per_sector);
    }
    LOG(INFO) << "Disk-Index File Meta-data: "
              << "# nodes per sector: " << nnodes_per_sector
              << ", max node len (bytes): " << max_node_len
              << ", max node degree: " << max_degree;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(files, medoids_file, medoids, num_medoids,
                                  tmp_dim);
#else
    if (file_exists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);
#endif

      if (tmp_dim != 1) {
        std::stringstream stream;
        stream << "Error loading medoids file. Expected bin format of m times "
                  "1 vector of uint32_t."
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
#ifdef EXEC_ENV_OLS
      if (!files.fileExists(centroids_file)) {
#else
      if (!file_exists(centroids_file)) {
#endif
        LOG(INFO)
            << "Centroid data file not found. Using corresponding vectors "
               "for the medoids ";
        use_medoids_data_as_centroids();
      } else {
        size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
        diskann::load_aligned_bin<float>(files, centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
#else
        diskann::load_aligned_bin<float>(centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
#endif
        if (aligned_tmp_dim != aligned_dim || num_centroids != num_medoids) {
          std::stringstream stream;
          stream << "Error loading centroids data file. Expected bin format of "
                    "m times data_dim vector of float, where m is number of "
                    "medoids "
                    "in medoids file.";
          LOG(ERROR) << stream.str();
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }
      }
    } else {
      num_medoids = 1;
      medoids = new uint32_t[1];
      medoids[0] = (_u32) (medoid_id_on_file);
      use_medoids_data_as_centroids();
    }

    std::string norm_file =
        get_disk_index_max_base_norm_file(std::string(disk_index_file));

    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT) {
      _u64   dumr, dumc;
      float *norm_val;
      diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
      this->max_base_norm = norm_val[0];
      LOG_KNOWHERE_DEBUG_ << "Setting re-scaling factor of base vectors to "
                          << this->max_base_norm;
      delete[] norm_val;
    }
    return 0;
  }

#ifdef USE_BING_INFRA
  bool getNextCompletedRequest(const IOContext &ctx, size_t size,
                               int &completedIndex) {
    bool waitsRemaining = false;
    for (int i = 0; i < size; i++) {
      auto ithStatus = (*ctx.m_pRequestsStatus)[i];
      if (ithStatus == IOContext::Status::READ_SUCCESS) {
        completedIndex = i;
        return true;
      } else if (ithStatus == IOContext::Status::READ_WAIT) {
        waitsRemaining = true;
      }
    }
    completedIndex = -1;
    return waitsRemaining;
  }
#endif

  template<typename T>
  std::optional<float> PQFlashIndex<T>::init_thread_data(ThreadData<T> &data,
                                                         const T *query1) {
    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    auto  q_dim = this->data_dim;
    if (metric == diskann::Metric::INNER_PRODUCT) {
      // query_dim need to be specially treated when using IP
      q_dim--;
    }
    for (uint32_t i = 0; i < q_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
      data.scratch.aligned_query_T[i] = query1[i];
      query_norm += query1[i] * query1[i];
    }

    // if inner product, we laso normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT) {
      if (query_norm == 0) {
        return std::nullopt;
      }
      query_norm = std::sqrt(query_norm);
      data.scratch.aligned_query_T[this->data_dim - 1] = 0;
      data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      for (uint32_t i = 0; i < this->data_dim - 1; i++) {
        data.scratch.aligned_query_T[i] /= query_norm;
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }

    data.scratch.reset();
    return query_norm;
  }

  template<typename T>
  void PQFlashIndex<T>::brute_force_beam_search(
      ThreadData<T> &data, const float query_norm, const _u64 k_search,
      _s64 *indices, float *distances, const _u64 beam_width_parm, IOContext &ctx,
      QueryStats *stats, const knowhere::feder::diskann::FederResultUniq &feder,
      knowhere::BitsetView bitset_view) {
    auto     query_scratch = &(data.scratch);
    const T *query = data.scratch.aligned_query_T;
    auto beam_width = beam_width_parm * kBruteForceBeamWidthFactor;

    T *data_buf = query_scratch->coord_scratch;
    std::unordered_map<_u64, std::vector<_u64>> nodes_in_sectors_to_visit;
    std::vector<AlignedRead>                    frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    Timer                                io_timer, query_timer;
    knowhere::ResultMaxHeap<float, _u64> max_heap(k_search);
    // TODO: maybe we can pipeline here
    assert(bitset_view.size() == num_points);
    for (_u64 id = 0; id < num_points;) {
      // gathering information about reading which nodes and which sector
      if (!bitset_view.test(id)) {
        const _u64 sector_offset = get_node_sector_offset(id);
        if (nodes_in_sectors_to_visit.find(sector_offset) ==
            nodes_in_sectors_to_visit.end()) {
          while (id < num_points &&
                 get_node_sector_offset(id) == sector_offset) {
            if (!bitset_view.test(id)) {
              if (coord_cache.find(id) != coord_cache.end()) {
                float dist =
                    dist_cmp(query, coord_cache.at(id), (size_t) aligned_dim);
                max_heap.Push(dist, id);
              } else {
                nodes_in_sectors_to_visit[sector_offset].push_back(id);
              }
            }
            ++id;
          }
        } else {
          LOG_KNOWHERE_ERROR_ << "Should not be visited twice";
        }
      } else {
        ++id;
      }
      if (id < num_points && nodes_in_sectors_to_visit.size() < beam_width)
        continue;

      // perform I/O
      for (const auto &[sector_offset, ids_in_sectors] :
           nodes_in_sectors_to_visit) {
        frontier_read_reqs.emplace_back(
            sector_offset, read_len_for_node,
            sector_scratch + sector_scratch_idx * read_len_for_node);
        ++sector_scratch_idx;
        if (stats != nullptr) {
          stats->n_4k++;
          stats->n_ios++;
        }
      }
      io_timer.reset();
#ifdef USE_BING_INFRA
      reader->read(frontier_read_reqs, ctx, true);  // async reader windows.
#else
      reader->read(frontier_read_reqs, ctx);    // synchronous IO linux
#endif
      if (stats != nullptr) {
        stats->io_us += (double) io_timer.elapsed();
      }

      T *node_fp_coords_copy = data_buf;
      for (const auto &req : frontier_read_reqs) {
        const _u64 sector_offset = req.offset;
        char      *sector_buf = reinterpret_cast<char *>(req.buf);
        for (const auto cur_id : nodes_in_sectors_to_visit[sector_offset]) {
          char *node_buf = get_offset_to_node(sector_buf, cur_id);
          memcpy(node_fp_coords_copy, node_buf,
                 disk_bytes_per_point);  // Do we really need memcpy here?
          float dist =
              dist_cmp(query, node_fp_coords_copy, (size_t) aligned_dim);
          max_heap.Push(dist, cur_id);
          if (feder != nullptr) {
            feder->visit_info_.AddTopCandidateInfo(cur_id, dist);
            feder->id_set_.insert(cur_id);
          }
        }
      }

      nodes_in_sectors_to_visit.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;
    }

    for (_s64 i = k_search - 1; i >= 0; --i) {
      if ((_u64) i >= max_heap.Size()) {
        indices[i] = -1;
        if (distances != nullptr) {
          distances[i] = -1;
        }
        continue;
      }
      if (const auto op = max_heap.Pop()) {
        const auto [dis, id] = op.value();
        indices[i] = id;
        if (distances != nullptr) {
          distances[i] = dis;
          if (metric == diskann::Metric::INNER_PRODUCT) {
            distances[i] = 1.0 - distances[i] / 2.0;
            if (max_base_norm != 0) {
              distances[i] *= (max_base_norm * query_norm);
            }
          }
        }
      } else {
        LOG_KNOWHERE_ERROR_ << "Size is incorrect";
      }
    }
    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return;
  }

  template<typename T>
  void PQFlashIndex<T>::cached_beam_search(
      const T *query1, const _u64 k_search, const _u64 l_search, _s64 *indices,
      float *distances, const _u64 beam_width, const bool use_reorder_data,
      QueryStats *stats, const knowhere::feder::diskann::FederResultUniq &feder,
      knowhere::BitsetView bitset_view) {
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    auto query_norm_opt = init_thread_data(data, query1);
    if (!query_norm_opt.has_value()) {
        // return an empty answer when calcu a zero point
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        return;
    }
    float query_norm = query_norm_opt.value();
    auto ctx = this->reader->get_ctx();

    if (!bitset_view.empty() && bitset_view.count() >= bitset_view.size() * kDiskAnnBruteForceFilterRate) {
        brute_force_beam_search(data, query_norm, k_search, indices, distances,
                                beam_width, ctx, stats, feder, bitset_view);
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        this->reader->put_ctx(ctx);
        return;
    }

    auto query_scratch = &(data.scratch);
    const T     *query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    Timer io_timer, query_timer;
    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8   *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };
    Timer                 cpu_timer;
    std::vector<Neighbor> retset(l_search + 1);
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);
    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist =
          dist_cmp_float(query_float, centroid_data + aligned_dim * cur_m,
                         (size_t) aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset[0].id = best_medoid;
    retset[0].flag = true;
    retset[0].distance = dist_scratch[0];
    visited.insert(best_medoid);

    unsigned cur_list_size = 1;

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;
      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width &&
             num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
          }
          retset[marker].flag = false;
          if (this->count_visited_nodes) {
            reinterpret_cast<std::atomic<_u32> &>(
                this->node_visit_counter[retset[marker].id].second)
                .fetch_add(1);
          }
          if (!bitset_view.empty() && bitset_view.test(retset[marker].id)) {
            std::memmove(&retset[marker], &retset[marker + 1],
                         (cur_list_size - marker - 1) * sizeof(Neighbor));
            cur_list_size--;
          } else {
            marker++;
          }
        } else {
          marker++;
        }
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto                    id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second =
              sector_scratch + sector_scratch_idx * read_len_for_node;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(get_node_sector_offset(((size_t) id)),
                                          read_len_for_node, fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(frontier_read_reqs, ctx, true);  // async reader windows.
#else
        reader->read(frontier_read_reqs, ctx);  // synchronous IO linux
#endif
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
      }

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto global_cache_iter = coord_cache.find(cached_nhood.first);
        T   *node_fp_coords_copy = global_cache_iter->second;
        if (bitset_view.empty() || !bitset_view.test(cached_nhood.first)) {
          float cur_expanded_dist;
          if (!use_disk_index_pq) {
            cur_expanded_dist =
                dist_cmp(query, node_fp_coords_copy, (size_t) aligned_dim);
          } else {
            if (metric == diskann::Metric::INNER_PRODUCT)
              cur_expanded_dist = disk_pq_table.inner_product(
                  query_float, (_u8 *) node_fp_coords_copy);
            else
              cur_expanded_dist = disk_pq_table.l2_distance(
                  query_float, (_u8 *) node_fp_coords_copy);
          }
          full_retset.push_back(
              Neighbor((unsigned) cached_nhood.first, cur_expanded_dist, true));

          // add top candidate info into feder result
          if (feder != nullptr) {
            feder->visit_info_.AddTopCandidateInfo(cached_nhood.first,
                                                   cur_expanded_dist);
            feder->id_set_.insert(cached_nhood.first);
          }
        }
        _u64      nnbrs = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

        // compute node_nbrs <-> query dists in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        // process prefetched nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];

          // add neighbor info into feder result
          if (feder != nullptr) {
            feder->visit_info_.AddTopCandidateNeighbor(cached_nhood.first, id,
                                                       dist_scratch[m]);
            feder->id_set_.insert(id);
          }

          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            if (cur_list_size > 0 &&
                dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            // Return position in sorted list where nn inserted.
            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < nk)
              // nk logs the best position in the retset that was
              // updated due to neighbors of n.
              nk = r;
          }
        }
      }
#ifdef USE_BING_INFRA
      // process each frontier nhood - compute distances to unvisited nodes
      int completedIndex = -1;
      // If we issued read requests and if a read is complete or there are reads
      // in wait state, then enter the while loop.
      while (frontier_read_reqs.size() > 0 &&
             getNextCompletedRequest(ctx, frontier_read_reqs.size(),
                                     completedIndex)) {
        if (completedIndex == -1) {  // all reads are waiting
          continue;
        }
        auto &frontier_nhood = frontier_nhoods[completedIndex];
        (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
      for (auto &frontier_nhood : frontier_nhoods) {
#endif
        char *node_disk_buf =
            get_offset_to_node(frontier_nhood.second, frontier_nhood.first);
        unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
        _u64      nnbrs = (_u64) (*node_buf);
        T        *node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);

        T *node_fp_coords_copy = data_buf;
        memcpy(node_fp_coords_copy, node_fp_coords, disk_bytes_per_point);
        if (bitset_view.empty() || !bitset_view.test(frontier_nhood.first)) {
          float cur_expanded_dist;
          if (!use_disk_index_pq) {
            cur_expanded_dist =
                dist_cmp(query, node_fp_coords_copy, (size_t) aligned_dim);
          } else {
            if (metric == diskann::Metric::INNER_PRODUCT)
              cur_expanded_dist = disk_pq_table.inner_product(
                  query_float, (_u8 *) node_fp_coords_copy);
            else
              cur_expanded_dist = disk_pq_table.l2_distance(
                  query_float, (_u8 *) node_fp_coords_copy);
          }
          full_retset.push_back(
              Neighbor(frontier_nhood.first, cur_expanded_dist, true));

          // add top candidate info into feder result
          if (feder != nullptr) {
            feder->visit_info_.AddTopCandidateInfo(frontier_nhood.first,
                                                   cur_expanded_dist);
            feder->id_set_.insert(frontier_nhood.first);
          }
        }
        unsigned *node_nbrs = (node_buf + 1);
        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];

          // add neighbor info into feder result
          if (feder != nullptr) {
            feder->visit_info_.AddTopCandidateNeighbor(frontier_nhood.first, id,
                                                       dist_scratch[m]);
            feder->id_set_.insert(frontier_nhood.first);
          }

          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (cur_list_size > 0 &&
                dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            auto     r = InsertIntoPool(
                    retset.data(), cur_list_size,
                    nn);  // Return position in sorted list where nn inserted.
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
                       // updated due to neighbors of n.
          }
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    if (use_reorder_data) {
      if (!(this->reorder_data_exists)) {
        throw ANNException(
            "Requested use of reordering data which does not exist in index "
            "file",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }

      std::vector<AlignedRead> vec_read_reqs;

      if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
        full_retset.erase(
            full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
            full_retset.end());

      for (size_t i = 0; i < full_retset.size(); ++i) {
        vec_read_reqs.emplace_back(
            VECTOR_SECTOR_NO(((size_t) full_retset[i].id)) * SECTOR_LEN,
            SECTOR_LEN, sector_scratch + i * SECTOR_LEN);

        if (stats != nullptr) {
          stats->n_4k++;
          stats->n_ios++;
        }
      }

      io_timer.reset();
#ifdef USE_BING_INFRA
      reader->read(vec_read_reqs, ctx, false);  // sync reader windows.
#else
      reader->read(vec_read_reqs, ctx);  // synchronous IO linux
#endif
      if (stats != nullptr) {
        stats->io_us += io_timer.elapsed();
      }

      for (size_t i = 0; i < full_retset.size(); ++i) {
        auto id = full_retset[i].id;
        auto location =
            (sector_scratch + i * SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
        full_retset[i].distance =
            dist_cmp(query, (T *) location, this->data_dim);
      }

      std::sort(full_retset.begin(), full_retset.end(),
                [](const Neighbor &left, const Neighbor &right) {
                  return left.distance < right.distance;
                });
    }

    // copy k_search values
    for (_u64 i = 0; i < k_search; i++) {
      if (i >= full_retset.size()) {
        indices[i] = -1;
        if (distances != nullptr) {
          distances[i] = -1;
        }
        continue;
      }
      indices[i] = full_retset[i].id;
      if (distances != nullptr) {
        distances[i] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // convert l2 distance to ip distance
          distances[i] = 1.0 - distances[i] / 2.0;
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[i] *= (max_base_norm * query_norm);
        }
      }
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);
    // std::cout << num_ios << " " <<stats << std::endl;

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  // range search returns results of all neighbors within distance of range.
  // indices and distances need to be pre-allocated of size l_search and the
  // return value is the number of matching hits.
  template<typename T>
  _u32 PQFlashIndex<T>::range_search(
      const T *query1, const double range, const _u64 min_l_search,
      const _u64 max_l_search, std::vector<_s64> &indices,
      std::vector<float> &distances, const _u64 beam_width,
      const float l_k_ratio, knowhere::BitsetView bitset_view,
      QueryStats *stats) {
    _u32 res_count = 0;

    bool stop_flag = false;

    _u32 l_search = min_l_search;  // starting size of the candidate list
    while (!stop_flag) {
      indices.resize(l_search);
      distances.resize(l_search);
      for (auto &x : distances)
        x = std::numeric_limits<float>::max();
      this->cached_beam_search(query1, l_search, l_k_ratio * l_search,
                               indices.data(), distances.data(), beam_width,
                               false, stats, nullptr, bitset_view);
      for (_u32 i = 0; i < l_search; i++) {
        if (indices[i] == -1) {
          res_count = i;
          break;
        }
        bool out_of_range = metric == diskann::Metric::INNER_PRODUCT
                                ? distances[i] < (float) range
                                : distances[i] > (float) range;
        if (out_of_range) {
          res_count = i;
          break;
        }
        if (i == l_search - 1) {
          res_count = l_search;
        }
      }
      if (res_count < (_u32) (l_search / 2.0))
        stop_flag = true;
      l_search = l_search * 2;
      if (l_search > max_l_search)
        stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
  }

  template<typename T>
  inline void
  PQFlashIndex<T>::copy_vec_base_data(T* des, const int64_t des_idx, void* src) {
    if (metric == Metric::INNER_PRODUCT) {
      assert(max_base_norm != 0);
      const auto original_dim = data_dim - 1;
      memcpy(des + des_idx * original_dim, src, original_dim * sizeof(T));
      for (size_t i = 0; i < original_dim; ++i) {
        des[des_idx * original_dim + i] *= max_base_norm;
      }
    } else {
      memcpy(des + des_idx * data_dim, src, data_dim * sizeof(T));
    }
  }

  template<typename T>
  std::unordered_map<_u64, std::vector<_u64>>
  PQFlashIndex<T>::get_sectors_layout_and_write_data_from_cache(const int64_t* ids, int64_t n, T* output_data) {
    std::unordered_map<_u64, std::vector<_u64>> sectors_to_visit;
    for (int64_t i = 0; i < n; ++i) {
      _u64 id = ids[i];
      if (coord_cache.find(id) != coord_cache.end()) {
        copy_vec_base_data(output_data, i, coord_cache.at(id));
      } else {
        const _u64 sector_offset = get_node_sector_offset(id);
        sectors_to_visit[sector_offset].push_back(i);
      }
    }
    return sectors_to_visit;
  }

  template<typename T>
  void PQFlashIndex<T>::get_vector_by_sector(const _u64 sector_offset,
                                             const std::vector<_u64> &ids_idx,
                                             const int64_t *ids, T *output_data) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    auto  ctx = this->reader->get_ctx();
    char *sector_scratch = data.scratch.sector_scratch;
    _u64 &sector_scratch_idx = data.scratch.sector_idx;
    std::vector<AlignedRead> frontier_read_reqs;

    char *sector_buf = sector_scratch + sector_scratch_idx * read_len_for_node;
    frontier_read_reqs.emplace_back(sector_offset, read_len_for_node,
                                    sector_buf);
#ifdef USE_BING_INFRA
    reader->read(frontier_read_reqs, ctx, true);  // async reader windows.
#else
    reader->read(frontier_read_reqs, ctx);  // synchronous IO linux
#endif
    for (const auto idx : ids_idx) {
      char *node_buf = get_offset_to_node(sector_buf, ids[idx]);
      copy_vec_base_data(output_data, idx, node_buf);
    }
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);
  }

  template<typename T>
  _u64 PQFlashIndex<T>::get_num_points() const noexcept {
    return num_points;
  }

  template<typename T>
  _u64 PQFlashIndex<T>::get_data_dim() const noexcept {
    return data_dim;
  }

  template<typename T>
  _u64 PQFlashIndex<T>::get_max_degree() const noexcept {
    return max_degree;
  }

  template<typename T>
  _u32 *PQFlashIndex<T>::get_medoids() const noexcept {
    return medoids;
  }

  template<typename T>
  size_t PQFlashIndex<T>::get_num_medoids() const noexcept {
    return num_medoids;
  }

  template<typename T>
  diskann::Metric PQFlashIndex<T>::get_metric() const noexcept {
    return metric;
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  char *PQFlashIndex<T>::getHeaderBytes() {
    IOContext  &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *) readReq.buf;
  }
#endif

  // instantiations
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;

}  // namespace diskann
