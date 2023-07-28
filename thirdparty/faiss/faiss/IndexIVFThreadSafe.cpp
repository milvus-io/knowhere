// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License

#include <faiss/IndexIVF.h>

#include <faiss/utils/utils.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <omp.h>
#include <cinttypes>
namespace faiss {

namespace {
IVFSearchParameters gen_search_param(
        const size_t& nprobe,
        const int parallel_mode,
        const size_t& max_codes) {
    IVFSearchParameters params;
    params.nprobe = nprobe;
    params.max_codes = max_codes;
    params.parallel_mode = parallel_mode;
    return params;
}
} // namespace

void IndexIVF::search_thread_safe(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const size_t nprobe,
        const int parallel_mode,
        const size_t max_codes,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);
    const size_t final_nprobe = std::min(nlist, nprobe);
    FAISS_THROW_IF_NOT(final_nprobe > 0);
    IVFSearchParameters params =
            gen_search_param(final_nprobe, parallel_mode, max_codes);

    // search function for a subset of queries
    auto sub_search_func = [this, k, final_nprobe, bitset, &params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * final_nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * final_nprobe]);

        double t0 = getmillisecs();
        quantizer->search(n, x, final_nprobe, coarse_dis.get(), idx.get());

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * final_nprobe);

        search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                &params,
                ivf_stats,
                bitset);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle paralellization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVF::search_without_codes_thread_safe(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const size_t nprobe,
        const int parallel_mode,
        const size_t max_codes,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);
    const size_t final_nprobe = std::min(nlist, nprobe);
    FAISS_THROW_IF_NOT(final_nprobe > 0);
    IVFSearchParameters params =
            gen_search_param(final_nprobe, parallel_mode, max_codes);

    // search function for a subset of queries
    auto sub_search_func = [this, k, final_nprobe, bitset, &params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * final_nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * final_nprobe]);

        double t0 = getmillisecs();
        quantizer->search(n, x, final_nprobe, coarse_dis.get(), idx.get());

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * final_nprobe);

        search_preassigned_without_codes(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                &params,
                ivf_stats,
                bitset);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle paralellization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVF::search_preassigned_without_codes(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    idx_t max_codes = params ? params->max_codes : this->max_codes;

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int preassigned_parallel_mode = 0;
    if (params && params->parallel_mode != -1) {
        preassigned_parallel_mode = params->parallel_mode;
    } else {
        preassigned_parallel_mode = this->parallel_mode;
    }
    int pmode = preassigned_parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init =
            !(preassigned_parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1);

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        InvertedListScanner* scanner = get_InvertedListScanner(store_pairs);
        ScopeDeleter1<InvertedListScanner> del(scanner);

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // intialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 const BitsetView bitset) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            size_t list_size = invlists->list_size(key);
            size_t offset = prefix_sum[key];

            // don't waste time on empty lists
            if (list_size == 0) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
#ifdef USE_GPU
                auto rol = dynamic_cast<faiss::ReadOnlyArrayInvertedLists*>(
                        invlists);
                auto arranged_data = reinterpret_cast<uint8_t*>(
                        rol->pin_readonly_codes->data);
                InvertedLists::ScopedCodes scodes(invlists, key, arranged_data);
#else
                InvertedLists::ScopedCodes scodes(
                        invlists, key, arranged_codes.data());
#endif

                std::unique_ptr<InvertedLists::ScopedIds> sids;
                const Index::idx_t* ids = nullptr;

                if (!store_pairs) {
                    sids.reset(new InvertedLists::ScopedIds(invlists, key));
                    ids = sids->get();
                }

                nheap += scanner->scan_codes(
                        list_size,
                        (scodes.get() + code_size * offset),
                        nullptr,
                        ids,
                        simi,
                        idxi,
                        k,
                        bitset);

            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }

            return list_size;
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            bitset);

                    if (max_codes && nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(),
                            local_idx.data(),
                            bitset);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;
                size_t j = ij % nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data(),
                        bitset);
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats) {
        ivf_stats->nq += n;
        ivf_stats->nlist += nlistv;
        ivf_stats->ndis += ndis;
        ivf_stats->nheap_updates += nheap;
    }
}

void IndexIVF::range_search_thread_safe(
        idx_t nx,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const size_t nprobe,
        const size_t max_codes,
        const BitsetView bitset) const {
    const size_t final_nprobe = std::min(nlist, nprobe);
    std::unique_ptr<idx_t[]> keys(new idx_t[nx * final_nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nx * final_nprobe]);

    double t0 = getmillisecs();
    quantizer->search(nx, x, final_nprobe, coarse_dis.get(), keys.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(keys.get(), nx * final_nprobe);

    IVFSearchParameters params = gen_search_param(final_nprobe, 0, max_codes);

    range_search_preassigned(
            nx,
            x,
            radius,
            keys.get(),
            coarse_dis.get(),
            result,
            false,
            &params,
            &indexIVF_stats,
            bitset);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIVF::range_search_without_codes_thread_safe(
        idx_t nx,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const size_t nprobe,
        const size_t max_codes,
        const BitsetView bitset) const {
    const size_t final_nprobe = std::min(nlist, nprobe);
    std::unique_ptr<idx_t[]> keys(new idx_t[nx * final_nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nx * final_nprobe]);

    double t0 = getmillisecs();
    quantizer->search(nx, x, final_nprobe, coarse_dis.get(), keys.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(keys.get(), nx * final_nprobe);

    IVFSearchParameters params = gen_search_param(final_nprobe, 0, max_codes);

    range_search_preassigned_without_codes(
            nx,
            x,
            radius,
            keys.get(),
            coarse_dis.get(),
            result,
            false,
            &params,
            &indexIVF_stats,
            bitset);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIVF::range_search_preassigned_without_codes(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats,
        const BitsetView bitset) const {
    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : this->max_codes;

    size_t nlistv = 0, ndis = 0;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

    int preassigned_parallel_mode = 0;
    if (params && params->parallel_mode != -1) {
        preassigned_parallel_mode = params->parallel_mode;
    } else {
        preassigned_parallel_mode = this->parallel_mode;
    }
    int pmode = preassigned_parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    // don't start parallel section if single query
    bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 3           ? false
                     : pmode == 0 ? nx > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * nx > 1);

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(result);
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs));
        FAISS_THROW_IF_NOT(scanner.get());
        all_pres[omp_get_thread_num()] = &pres;

        // prepare the list scanning function

        auto scan_list_func = [&](size_t i,
                                  size_t ik,
                                  RangeQueryResult& qres,
                                  const BitsetView bitset) {
            idx_t key = keys[i * nprobe + ik]; /* select the list  */
            if (key < 0)
                return;
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    nlist);
            const size_t list_size = invlists->list_size(key);
            const size_t offset = prefix_sum[key];

            if (list_size == 0)
                return;

            try {
#ifdef USE_GPU
                auto rol = dynamic_cast<faiss::ReadOnlyArrayInvertedLists*>(
                        invlists);
                auto arranged_data = reinterpret_cast<uint8_t*>(
                        rol->pin_readonly_codes->data);
                InvertedLists::ScopedCodes scodes(invlists, key, arranged_data);
#else
                InvertedLists::ScopedCodes scodes(
                        invlists, key, arranged_codes.data());
#endif
                InvertedLists::ScopedIds ids(invlists, key);

                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                nlistv++;
                ndis += list_size;
                scanner->scan_codes_range(
                        list_size,
                        (scodes.get() + code_size * offset),
                        nullptr,
                        ids.get(),
                        radius,
                        qres,
                        bitset);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
            }
        };

#pragma omp for
        for (idx_t i = 0; i < nx; i++) {
            scanner->set_query(x + i * d);

            RangeQueryResult& qres = pres.new_result(i);
            size_t prev_nres = qres.nres;

            for (size_t ik = 0; ik < nprobe; ik++) {
                scan_list_func(i, ik, qres, bitset);
                if (qres.nres == prev_nres) break;
                prev_nres = qres.nres;
            }
        }
        pres.finalize();
    }

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (stats) {
        stats->nq += nx;
        stats->nlist += nlistv;
        stats->ndis += ndis;
    }
}

} // namespace faiss
