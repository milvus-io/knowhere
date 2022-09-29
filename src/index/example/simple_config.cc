#include "index/example/simple_config.h"

#include <queue>

#include "knowhere/knowhere.h"

namespace knowhere {

class SimpleIndexNode : public IndexNode {
 public:
    SimpleIndexNode() : xb(nullptr), dim(0), nb(0) {
    }
    Error
    Build(const DataSet& dataset, const Config& cfg);
    Error
    Train(const DataSet& dataset, const Config& cfg);
    Error
    Add(const DataSet& dataset, const Config& cfg);
    expected<DataSetPtr, Error>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const;
    expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const;
    Error
    Serialization(BinarySet& binset) const {
        return Error::not_implemented;
    };
    Error
    Deserialization(const BinarySet& binset) {
        return Error::not_implemented;
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

Error
SimpleIndexNode::Build(const DataSet& dataset, const Config& cfg) {
    auto err = this->Train(dataset, cfg);
    if (err != Error::success)
        return err;
    return this->Add(dataset, cfg);
}

Error
SimpleIndexNode::Train(const DataSet& dataset, const Config& cfg) {
    return Error::success;
}

Error
SimpleIndexNode::Add(const DataSet& dataset, const Config& cfg) {
    this->xb = (float*)dataset.GetTensor();
    const SimpleConfig& s_cfg = static_cast<const SimpleConfig&>(cfg);
    this->dim = s_cfg.dim;
    this->nb = dataset.GetRows();
    return Error::success;
}

expected<DataSetPtr, Error>
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

expected<DataSetPtr, Error>
SimpleIndexNode::GetVectorByIds(const DataSet& dataset, const Config& cfg) const {
    return unexpected(Error::not_implemented);
}

KNOWHERE_REGISTER_GLOBAL(SIMPLEINDEX, []() { return Index<SimpleIndexNode>(); });
}  // namespace knowhere
