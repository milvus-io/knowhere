#ifndef INDEX_TYPE_H
#define INDEX_TYPE_H

#include <string>

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

enum class IndexMode { MODE_CPU = 0, MODE_GPU = 1 };

}  // namespace knowhere

#endif /* INDEX_TYPE_H */
