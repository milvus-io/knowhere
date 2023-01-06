#include "knowhere/feder/HNSW.h"

#include <omp.h>

#include "common/range_util.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "index/hnsw/hnsw_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/factory.h"

namespace knowhere {
class HnswIndexNode : public IndexNode {
 public:
    HnswIndexNode(const Object& object) : index_(nullptr) {
        pool_ = ThreadPool::GetGlobalThreadPool();
    }

    Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto res = this->Train(dataset, cfg);
        if (res != Status::success) {
            return res;
        }
        res = Add(dataset, cfg);
        return res;
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        hnswlib::SpaceInterface<float>* space = nullptr;
        if (hnsw_cfg.metric_type == metric::L2) {
            space = new (std::nothrow) hnswlib::L2Space(dim);
        } else if (hnsw_cfg.metric_type == metric::IP) {
            space = new (std::nothrow) hnswlib::InnerProductSpace(dim);
        } else {
            LOG_KNOWHERE_WARNING_ << "metric type not support in hnsw, " << hnsw_cfg.metric_type;
            return Status::invalid_metric_type;
        }
        auto index =
            new (std::nothrow) hnswlib::HierarchicalNSW<float>(space, rows, hnsw_cfg.M, hnsw_cfg.efConstruction);
        if (index == nullptr) {
            LOG_KNOWHERE_WARNING_ << "memory malloc error.";
            return Status::malloc_error;
        }
        if (this->index_) {
            delete this->index_;
            LOG_KNOWHERE_WARNING_ << "index not empty, deleted old index.";
        }
        this->index_ = index;
        return Status::success;
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        if (!index_) {
            return Status::empty_index;
        }

        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto tensor = dataset.GetTensor();
        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        index_->addPoint(tensor, 0);

#pragma omp parallel for
        for (int i = 1; i < rows; ++i) {
            index_->addPoint((static_cast<const float*>(tensor) + dim * i), i);
        }
        return Status::success;
    }

    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return unexpected(Status::empty_index);
        }

        auto nq = dataset.GetRows();
        auto dim = dataset.GetDim();
        const float* xq = static_cast<const float*>(dataset.GetTensor());

        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        auto k = hnsw_cfg.k;

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit) {
            if (nq != 1) {
                return unexpected(Status::invalid_args);
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        auto p_id = new int64_t[k * nq];
        auto p_dist = new float[k * nq];

        hnswlib::SearchParam param{(size_t)hnsw_cfg.ef};
        bool transform = (index_->metric_type_ == 1);  // InnerProduct: 1

        std::vector<std::future<void>> futures;
        futures.reserve(nq);
        for (int i = 0; i < nq; ++i) {
            futures.push_back(pool_->push([&, index = i]() {
                auto single_query = xq + index * dim;
                auto rst = index_->searchKnn(single_query, k, bitset, &param, feder_result);
                size_t rst_size = rst.size();
                auto p_single_dis = p_dist + index * k;
                auto p_single_id = p_id + index * k;
                for (size_t idx = 0; idx < rst_size; ++idx) {
                    const auto& [dist, id] = rst[idx];
                    p_single_dis[idx] = transform ? (1 - dist) : dist;
                    p_single_id[idx] = id;
                }
                for (size_t idx = rst_size; idx < (size_t)k; idx++) {
                    p_single_dis[idx] = float(1.0 / 0.0);
                    p_single_id[idx] = -1;
                }
            }));
        }
        for (auto& future : futures) {
            future.get();
        }

        auto res = GenResultDataSet(nq, k, p_id, p_dist);

        // set visit_info json string into result dataset
        if (feder_result != nullptr) {
            Json json_visit_info, json_id_set;
            nlohmann::to_json(json_visit_info, feder_result->visit_info_);
            nlohmann::to_json(json_id_set, feder_result->id_set_);
            res->SetJsonInfo(json_visit_info.dump());
            res->SetJsonIdSet(json_id_set.dump());
        }
        return res;
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "range search on empty index.";
            return unexpected(Status::empty_index);
        }

        auto nq = dataset.GetRows();
        auto dim = dataset.GetDim();
        const float* xq = static_cast<const float*>(dataset.GetTensor());

        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        float low_bound = hnsw_cfg.radius_low_bound;
        float high_bound = hnsw_cfg.radius_high_bound;
        bool is_ip = (index_->metric_type_ == 1);  // 0:L2, 1:IP
        float radius = (is_ip ? (1.0f - low_bound) : high_bound);

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit) {
            if (nq != 1) {
                return unexpected(Status::invalid_args);
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        hnswlib::SearchParam param{(size_t)hnsw_cfg.ef};

        int64_t* ids = nullptr;
        float* dis = nullptr;
        size_t* lims = nullptr;

        std::vector<std::vector<int64_t>> result_id_array(nq);
        std::vector<std::vector<float>> result_dist_array(nq);
        std::vector<size_t> result_size(nq);
        std::vector<size_t> result_lims(nq + 1);

        std::vector<std::future<void>> futures;
        futures.reserve(nq);
        for (int64_t i = 0; i < nq; ++i) {
            futures.push_back(pool_->push([&, index = i]() {
                auto single_query = xq + index * dim;
                auto rst = index_->searchRange(single_query, radius, bitset, &param, feder_result);
                auto elem_cnt = rst.size();
                result_dist_array[index].resize(elem_cnt);
                result_id_array[index].resize(elem_cnt);
                for (size_t j = 0; j < elem_cnt; j++) {
                    auto& p = rst[j];
                    result_dist_array[index][j] = (is_ip ? (1 - p.first) : p.first);
                    result_id_array[index][j] = p.second;
                }
                result_size[index] = rst.size();
                FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, low_bound,
                                                high_bound);
            }));
        }
        for (auto& future : futures) {
            future.get();
        }

        // filter range search result
        GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, low_bound, high_bound, dis, ids, lims);

        auto res = GenResultDataSet(nq, ids, dis, lims);

        // set visit_info json string into result dataset
        if (feder_result != nullptr) {
            Json json_visit_info, json_id_set;
            nlohmann::to_json(json_visit_info, feder_result->visit_info_);
            nlohmann::to_json(json_id_set, feder_result->id_set_);
            res->SetJsonInfo(json_visit_info.dump());
            res->SetJsonIdSet(json_id_set.dump());
        }
        return res;
    }

    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        if (!index_) {
            return unexpected(Status::empty_index);
        }

        auto dim = dataset.GetDim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();

        float* p_x = nullptr;
        try {
            p_x = new float[dim * rows];
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < (int64_t)index_->cur_element_count);
                memcpy(p_x + i * dim, index_->getDataByInternalId(id), dim * sizeof(float));
            }
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error, " << e.what();
            std::unique_ptr<float> auto_delete_px(p_x);
            return unexpected(Status::hnsw_inner_error);
        }
        return GenResultDataSet(p_x);
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "get index meta on empty index.";
            return unexpected(Status::empty_index);
        }

        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        auto overview_levels = hnsw_cfg.overview_levels;
        feder::hnsw::HNSWMeta meta(index_->ef_construction_, index_->M_, index_->cur_element_count, index_->maxlevel_,
                                   index_->enterpoint_node_, overview_levels);
        std::unordered_set<int64_t> id_set;

        for (int i = 0; i < overview_levels; i++) {
            int64_t level = index_->maxlevel_ - i;
            // do not record level 0
            if (level <= 0) {
                break;
            }
            meta.AddLevelLinkGraph(level);
            UpdateLevelLinkList(level, meta, id_set);
        }

        Json json_meta, json_id_set;
        nlohmann::to_json(json_meta, meta);
        nlohmann::to_json(json_id_set, id_set);
        return GenResultDataSet(json_meta.dump(), json_id_set.dump());
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            index_->saveIndex(writer);
            std::shared_ptr<uint8_t[]> data(writer.data_);
            binset.Append("HNSW", data, writer.rp);
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error, " << e.what();
            return Status::hnsw_inner_error;
        }
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset) override {
        if (index_) {
            delete index_;
        }
        try {
            auto binary = binset.GetByName("HNSW");

            MemoryIOReader reader;
            reader.total = binary->size;
            reader.data_ = binary->data.get();

            hnswlib::SpaceInterface<float>* space = nullptr;
            index_ = new (std::nothrow) hnswlib::HierarchicalNSW<float>(space);
            index_->loadIndex(reader);
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error, " << e.what();
            return Status::hnsw_inner_error;
        }
        return Status::success;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<HnswConfig>();
    }

    int64_t
    Dim() const override {
        if (!index_) {
            return 0;
        }
        return (*static_cast<size_t*>(index_->dist_func_param_));
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
        return index_->cur_element_count;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_HNSW;
    }

    ~HnswIndexNode() override {
        if (index_) {
            delete index_;
        }
    }

 private:
    void
    UpdateLevelLinkList(int32_t level, feder::hnsw::HNSWMeta& meta, std::unordered_set<int64_t>& id_set) const {
        if (!(level > 0 && level <= index_->maxlevel_)) {
            return;
        }
        if (index_->cur_element_count == 0) {
            return;
        }

        std::vector<hnswlib::tableint> level_elements;

        // get all elements in current level
        for (size_t i = 0; i < index_->cur_element_count; i++) {
            // elements in high level also exist in low level
            if (index_->element_levels_[i] >= level) {
                level_elements.emplace_back(i);
            }
        }

        // iterate all elements in current level, record their link lists
        for (auto curr_id : level_elements) {
            auto data = index_->get_linklist(curr_id, level);
            auto size = index_->getListCount(data);

            hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
            std::vector<int64_t> neighbors(size);
            for (int i = 0; i < size; i++) {
                hnswlib::tableint cand = datal[i];
                neighbors[i] = cand;
            }
            id_set.insert(curr_id);
            id_set.insert(neighbors.begin(), neighbors.end());
            meta.AddNodeInfo(level, curr_id, std::move(neighbors));
        }
    }

 private:
    hnswlib::HierarchicalNSW<float>* index_;
    std::shared_ptr<ThreadPool> pool_;
};

KNOWHERE_REGISTER_GLOBAL(HNSW, [](const Object& object) { return Index<HnswIndexNode>::Create(object); });

}  // namespace knowhere
