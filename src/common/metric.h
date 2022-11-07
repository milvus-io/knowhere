#ifndef COMMON_METRIC_H
#define COMMON_METRIC_H

#include <algorithm>
#include <string>
#include <unordered_map>

#include "faiss/MetricType.h"
#include "knowhere/expected.h"
namespace knowhere {

inline expected<faiss::MetricType, Status>
Str2FaissMetricType(std::string metric) {
    static const std::unordered_map<std::string, faiss::MetricType> metric_map = {
        {"L2", faiss::MetricType::METRIC_L2},
        {"IP", faiss::MetricType::METRIC_INNER_PRODUCT},
        {"JACCARD", faiss::MetricType::METRIC_Jaccard},
        {"TANIMOTO", faiss::MetricType::METRIC_Tanimoto},
        {"HAMMING", faiss::MetricType::METRIC_Hamming},
        {"SUBSTRUCTURE", faiss::MetricType::METRIC_Substructure},
        {"SUPERSTRUCTURE", faiss::MetricType::METRIC_Superstructure},
    };

    std::transform(metric.begin(), metric.end(), metric.begin(), toupper);
    auto it = metric_map.find(metric);
    if (it == metric_map.end())
        return unexpected(Status::invalid_metric_type);
    return it->second;
}

}  // namespace knowhere

#endif /* METRIC_H */
