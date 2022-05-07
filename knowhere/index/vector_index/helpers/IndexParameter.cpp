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

#include "common/Exception.h"
#include "index/vector_index/helpers/IndexParameter.h"

namespace knowhere {

faiss::MetricType
GetMetricType(const std::string& type) {
    if (type == metric::L2) {
        return faiss::METRIC_L2;
    } else if (type == metric::IP) {
        return faiss::METRIC_INNER_PRODUCT;
    } else if (type == metric::JACCARD) {
        return faiss::METRIC_Jaccard;
    } else if (type == metric::TANIMOTO) {
        return faiss::METRIC_Tanimoto;
    } else if (type == metric::HAMMING) {
        return faiss::METRIC_Hamming;
    } else if (type == metric::SUBSTRUCTURE) {
        return faiss::METRIC_Substructure;
    } else if (type == metric::SUPERSTRUCTURE) {
        return faiss::METRIC_Superstructure;
    } else {
        KNOWHERE_THROW_MSG("Metric type invalid");
    }
}

}  // namespace knowhere
