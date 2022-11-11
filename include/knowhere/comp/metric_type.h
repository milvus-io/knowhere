#pragma once

#include <string>

namespace knowhere {

using MetricType = std::string;

namespace metric {
constexpr const char* IP = "IP";
constexpr const char* L2 = "L2";
constexpr const char* HAMMING = "HAMMING";
constexpr const char* JACCARD = "JACCARD";
constexpr const char* TANIMOTO = "TANIMOTO";
constexpr const char* SUBSTRUCTURE = "SUBSTRUCTURE";
constexpr const char* SUPERSTRUCTURE = "SUPERSTRUCTURE";
}  // namespace metric

}  // namespace knowhere
