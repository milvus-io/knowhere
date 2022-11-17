#include "index/example/simple_config.h"

#include <queue>

#include "knowhere/knowhere.h"

namespace knowhere {

class SimpleIndexNode : public IndexNode {
 public:
    SimpleIndexNode(const Object& object) : xb(nullptr), dim(0), nb(0) {
    }
    Status
    Build(const DataSet& dataset, const Config& cfg);
    Status
    Train(const DataSet& dataset, const Config& cfg);
    Status
    Add(const DataSet& dataset, const Config& cfg);
    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const;
    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
        return unexpected(Status::not_implemented);
    }
    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const {
        return unexpected(Status::not_implemented);
    }
    Status
    Serialization(BinarySet& binset) const {
        return Status::not_implemented;
    };
    Status
    Deserialization(const BinarySet& binset) {
        return Status::not_implemented;
    };
    std::unique_ptr<BaseConfig>
    CreateConfig() const {
        return std::make_unique<SimpleConfig>();
    }
    int64_t
    Dims() const {
        return dim;
    }
    int64_t
    Size() const {
        return 0;
    }
    int64_t
    Count() const {
        return nb;
    }
    std::string
    Type() const {
        return "SIMPLEINDEX";
    }
    virtual ~SimpleIndexNode(){};

 private:
    float* xb;
    int dim;
    int nb;
};

Status
SimpleIndexNode::Build(const DataSet& dataset, const Config& cfg) {
    auto err = this->Train(dataset, cfg);
    if (err != Status::success)
        return err;
    return this->Add(dataset, cfg);
}

Status
SimpleIndexNode::Train(const DataSet& dataset, const Config& cfg) {
    return Status::success;
}

Status
SimpleIndexNode::Add(const DataSet& dataset, const Config& cfg) {
    this->xb = (float*)dataset.GetTensor();
    const SimpleConfig& s_cfg = static_cast<const SimpleConfig&>(cfg);
    this->dim = dataset.GetDim();
    this->nb = dataset.GetRows();
    return Status::success;
}

expected<DataSetPtr, Status>
SimpleIndexNode::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second < b.second; };
    float* xq = (float*)dataset.GetTensor();
    int nq = dataset.GetRows();
    const SimpleConfig& s_cfg = static_cast<const SimpleConfig&>(cfg);
    int top_k = s_cfg.k;
    auto len = top_k * nq;
    auto ids = new (std::nothrow) int64_t[len];
    auto dis = new (std::nothrow) float[len];
    auto ans = std::make_shared<DataSet>();
    for (int i = 0; i < nq; ++i) {
        std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, decltype(cmp)> top_q(cmp);
        float* a = xq + i * dim;

        for (int j = 0; j < nb; ++j) {
            float* b = xb + j * dim;
            float dis = 0.0f;
            for (int k = 0; k < dim; ++k) {
                if (s_cfg.metric_type == "L2")
                    dis += (a[k] - b[k]) * (a[k] - b[k]);
                if (s_cfg.metric_type == "L1")
                    dis += std::fabs(a[k] - b[k]);
            }
            top_q.emplace(j, dis);
        }
        for (int k = 0; k < top_k; ++k) {
            auto t = top_q.top();
            ids[i * top_k + k] = t.first;
            dis[i * top_k + k] = t.second;
            top_q.pop();
        }
    }
    ans->SetIds(ids);
    ans->SetDistance(dis);
    ans->SetDim(dim);
    ans->SetRows(nq);
    return ans;
}

KNOWHERE_REGISTER_GLOBAL(SIMPLEINDEX, [](const Object& object) { return Index<SimpleIndexNode>::Create(object); });
}  // namespace knowhere
