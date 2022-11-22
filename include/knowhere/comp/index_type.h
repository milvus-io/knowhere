#ifndef INDEX_TYPE_H
#define INDEX_TYPE_H

#include <string>

#include "knowhere/comp/MetricType.h"
namespace knowhere {

using IndexType = std::string;

namespace IndexEnum {

constexpr const char* INVALID = "";

constexpr const char* INDEX_FAISS_BIN_IDMAP = "BIN_FLAT";
constexpr const char* INDEX_FAISS_BIN_IVFFLAT = "BIN_IVF_FLAT";

constexpr const char* INDEX_FAISS_IDMAP = "FLAT";
constexpr const char* INDEX_FAISS_IVFFLAT = "IVF_FLAT";
constexpr const char* INDEX_FAISS_IVFPQ = "IVF_PQ";
constexpr const char* INDEX_FAISS_IVFSQ8 = "IVF_SQ8";

constexpr const char* INDEX_ANNOY = "ANNOY";
constexpr const char* INDEX_HNSW = "HNSW";

constexpr const char* INDEX_DISKANN = "DISKANN";

}  // namespace IndexEnum

namespace meta {
constexpr const char* TOPK = "k";
constexpr const char* METRIC_TYPE = "metric_type";
constexpr const char* RADIUS = "radius";
constexpr const char* INPUT_IDS = "input_ids";
constexpr const char* OUTPUT_TENSOR = "output_tensor";
constexpr const char* DEVICE_ID = "gpu_id";
constexpr const char* BUILD_INDEX_OMP_NUM = "build_index_omp_num";
constexpr const char* QUERY_OMP_NUM = "query_omp_num";
constexpr const char* TRACE_VISIT = "trace_visit";
constexpr const char* JSON_INFO = "json_info";
constexpr const char* JSON_ID_SET = "json_id_set";
};  // namespace meta

namespace indexparam {
// IVF Params
constexpr const char* NPROBE = "nprobe";
constexpr const char* NLIST = "nlist";
constexpr const char* NBITS = "nbits";  // PQ/SQ
constexpr const char* M = "m";          // PQ param for IVFPQ
constexpr const char* PQ_M = "PQM";     // PQ param for RHNSWPQ
// HNSW Params
constexpr const char* EFCONSTRUCTION = "efConstruction";
constexpr const char* HNSW_M = "M";
constexpr const char* EF = "ef";
constexpr const char* OVERVIEW_LEVELS = "overview_levels";
// Annoy Params
constexpr const char* N_TREES = "n_trees";
constexpr const char* SEARCH_K = "search_k";
}  // namespace indexparam

using MetricType = std::string;

// namespace metric

typedef std::string MetricType;
enum class IndexMode { MODE_CPU = 0, MODE_GPU = 1 };

}  // namespace knowhere

#endif /* INDEX_TYPE_H */
