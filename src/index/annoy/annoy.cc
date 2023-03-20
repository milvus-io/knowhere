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

#include "annoylib.h"
#include "index/annoy/annoy_config.h"
#include "kissrandom.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/expected.h"
#include "knowhere/factory.h"
namespace knowhere {

using ThreadedBuildPolicy = AnnoyIndexSingleThreadedBuildPolicy;
class AnnoyIndexNode : public IndexNode {
 public:
    AnnoyIndexNode(const Object& object) : index_(nullptr), pool_(ThreadPool::GetGlobalThreadPool()) {
    }

    Status
    Build(const DataSet& dataset, const Config& cfg) override {
        const AnnoyConfig& annoy_cfg = static_cast<const AnnoyConfig&>(cfg);
        metric_type_ = annoy_cfg.metric_type;
        auto dim = dataset.GetDim();
        AnnoyIndexInterface<int64_t, float>* index = nullptr;
        if (annoy_cfg.metric_type == "L2") {
            index =
                new (std::nothrow) AnnoyIndex<int64_t, float, ::Euclidean, ::Kiss64Random, ThreadedBuildPolicy>(dim);
            if (index == nullptr) {
                LOG_KNOWHERE_WARNING_ << "malloc memory error.";
                return Status::malloc_error;
            }
        }

        if (annoy_cfg.metric_type == "IP") {
            index =
                new (std::nothrow) AnnoyIndex<int64_t, float, ::DotProduct, ::Kiss64Random, ThreadedBuildPolicy>(dim);
            if (index == nullptr) {
                LOG_KNOWHERE_WARNING_ << "malloc memory error.";
                return Status::malloc_error;
            }
        }
        if (index) {
            if (this->index_) {
                delete this->index_;
            }
            this->index_ = index;

            auto p_data = dataset.GetTensor();
            auto rows = dataset.GetRows();
            for (int i = 0; i < rows; ++i) {
                index_->add_item(i, static_cast<const float*>(p_data) + dim * i);
            }
            char* error_msg;
            bool res = index_->build(annoy_cfg.n_trees, -1, &error_msg);
            if (!res) {
                LOG_KNOWHERE_WARNING_ << error_msg;
                free(error_msg);
                return Status::annoy_inner_error;
            }
            return Status::success;
        }

        LOG_KNOWHERE_WARNING_ << "invalid metric type in annoy " << annoy_cfg.metric_type;

        return Status::invalid_metric_type;
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        return this->Build(dataset, cfg);
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            return unexpected(Status::empty_index);
        }

        auto dim = dataset.GetDim();
        auto rows = dataset.GetRows();
        auto ts = dataset.GetTensor();
        auto annoy_cfg = static_cast<const AnnoyConfig&>(cfg);
        auto p_id = new (std::nothrow) int64_t[annoy_cfg.k * rows];
        auto p_dist = new (std::nothrow) float[annoy_cfg.k * rows];

        std::vector<std::future<void>> futures;
        futures.reserve(rows);
        for (unsigned int i = 0; i < rows; ++i) {
            futures.push_back(pool_->push([&, index = i]() {
                std::vector<int64_t> result;
                result.reserve(annoy_cfg.k);
                std::vector<float> distances;
                distances.reserve(annoy_cfg.k);
                index_->get_nns_by_vector(static_cast<const float*>(ts) + index * dim, annoy_cfg.k, annoy_cfg.search_k,
                                          &result, &distances, bitset);

                size_t result_num = result.size();
                auto local_p_id = p_id + annoy_cfg.k * index;
                auto local_p_dist = p_dist + annoy_cfg.k * index;
                memcpy(local_p_id, result.data(), result_num * sizeof(int64_t));
                memcpy(local_p_dist, distances.data(), result_num * sizeof(float));

                for (; result_num < (size_t)annoy_cfg.k; result_num++) {
                    local_p_id[result_num] = -1;
                    local_p_dist[result_num] = 1.0 / 0.0;
                }
            }));
        }

        for (auto& future : futures) {
            future.get();
        }

        return GenResultDataSet(rows, annoy_cfg.k, p_id, p_dist);
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return unexpected(Status::not_implemented);
    }

    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        if (!index_) {
            return unexpected(Status::empty_index);
        }

        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto p_ids = dataset.GetIds();

        float* p_x = nullptr;
        try {
            p_x = new (std::nothrow) float[dim * rows];
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = p_ids[i];
                assert(id >= 0 && id < index_->get_n_items());
                index_->get_item(id, p_x + i * dim);
            }
        } catch (const std::exception& e) {
            std::unique_ptr<float> auto_del(p_x);
            LOG_KNOWHERE_WARNING_ << "error in annoy, " << e.what();
            return unexpected(Status::annoy_inner_error);
        }

        return GenResultDataSet(p_x);
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            return Status::empty_index;
        }

        auto metric_type_length = metric_type_.length();
        std::shared_ptr<uint8_t[]> metric_type(new uint8_t[metric_type_length]);
        memcpy(metric_type.get(), metric_type_.data(), metric_type_.length());

        auto dim = Dim();
        std::shared_ptr<uint8_t[]> dim_data(new uint8_t[sizeof(uint64_t)]);
        memcpy(dim_data.get(), &dim, sizeof(uint64_t));

        size_t index_length = index_->get_index_length();
        std::shared_ptr<uint8_t[]> index_data(new uint8_t[index_length]);
        memcpy(index_data.get(), index_->get_index(), index_length);

        binset.Append("annoy_metric_type", metric_type, metric_type_length);
        binset.Append("annoy_dim", dim_data, sizeof(uint64_t));
        binset.Append("annoy_index_data", index_data, index_length);

        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset) override {
        if (index_) {
            delete index_;
        }
        auto metric_type = binset.GetByName("annoy_metric_type");
        metric_type_.resize(static_cast<size_t>(metric_type->size));
        memcpy(metric_type_.data(), metric_type->data.get(), static_cast<size_t>(metric_type->size));

        auto dim_data = binset.GetByName("annoy_dim");
        uint64_t dim;
        memcpy(&dim, dim_data->data.get(), static_cast<size_t>(dim_data->size));

        if (metric_type_ == "L2") {
            index_ =
                new (std::nothrow) AnnoyIndex<int64_t, float, ::Euclidean, ::Kiss64Random, ThreadedBuildPolicy>(dim);
        } else if (metric_type_ == "IP") {
            index_ =
                new (std::nothrow) AnnoyIndex<int64_t, float, ::DotProduct, ::Kiss64Random, ThreadedBuildPolicy>(dim);
        }

        auto index_data = binset.GetByName("annoy_index_data");
        char* p = nullptr;
        if (!index_->load_index(reinterpret_cast<void*>(index_data->data.get()), index_data->size, &p)) {
            free(p);
            return Status::annoy_inner_error;
        }

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, const LoadConfig& config) override {
        return Status::not_implemented;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<AnnoyConfig>();
    }

    int64_t
    Dim() const override {
        if (!index_) {
            return 0;
        }
        return index_->get_dim();
    }

    int64_t
    Size() const override {
        if (!index_) {
            return 0;
        }
        return index_->cal_size();
    }

    int64_t
    Count() const override {
        if (!index_) {
            return 0;
        }
        return index_->get_n_items();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_ANNOY;
    }

    ~AnnoyIndexNode() override {
        if (index_) {
            delete index_;
        }
    }

 private:
    AnnoyIndexInterface<int64_t, float>* index_;
    std::string metric_type_;
    std::shared_ptr<ThreadPool> pool_;
};

KNOWHERE_REGISTER_GLOBAL(ANNOY, [](const Object& object) { return Index<AnnoyIndexNode>::Create(object); });

}  // namespace knowhere
