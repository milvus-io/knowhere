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

#include "common/metric.h"
#include "faiss/IndexFlat.h"
#include "faiss/gpu/GpuCloner.h"
#include "faiss/index_io.h"
#include "index/flat_gpu/flat_gpu_config.h"
#include "index/gpu/gpu_res_mgr.h"
#include "io/FaissIO.h"
#include "knowhere/factory.h"

namespace knowhere {

class GpuFlatIndexNode : public IndexNode {
 public:
    GpuFlatIndexNode(const Object& object) : index_(nullptr) {
    }

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success)
            return err;
        err = Add(dataset, cfg);
        return err;
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) override {
        const GpuFlatConfig& f_cfg = static_cast<const GpuFlatConfig&>(cfg);
        auto metric = Str2FaissMetricType(f_cfg.metric_type);
        if (!metric.has_value()) {
            LOG_KNOWHERE_WARNING_ << "metric type error, " << f_cfg.metric_type;
            return metric.error();
        }
        index_ = std::make_unique<faiss::IndexFlat>(dataset.GetDim(), metric.value());
        return Status::success;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        const void* x = dataset.GetTensor();
        const int64_t n = dataset.GetRows();
        try {
            index_->add(n, (const float*)x);
            // need not copy index from CPU to GPU for IDMAP
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "index not empty, deleted old index.";
            return unexpected(Status::empty_index);
        }

        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        auto nq = dataset.GetRows();
        auto x = dataset.GetTensor();
        auto len = f_cfg.k * nq;
        int64_t* ids = nullptr;
        float* dis = nullptr;
        try {
            ids = new (std::nothrow) int64_t[len];
            dis = new (std::nothrow) float[len];

            ResScope rs(res_, false);
            index_->search(nq, (const float*)x, f_cfg.k, dis, ids, bitset);
        } catch (const std::exception& e) {
            std::unique_ptr<int64_t[]> auto_delete_ids(ids);
            std::unique_ptr<float[]> auto_delete_dis(dis);
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return unexpected(Status::faiss_inner_error);
        }

        return GenResultDataSet(nq, f_cfg.k, ids, dis);
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return unexpected(Status::not_implemented);
    }

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        DataSetPtr results = std::make_shared<DataSet>();
        auto nq = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto in_ids = dataset.GetIds();
        try {
            float* xq = new (std::nothrow) float[nq * dim];
            for (int64_t i = 0; i < nq; i++) {
                int64_t id = in_ids[i];
                index_->reconstruct(id, xq + i * dim);
            }
            return GenResultDataSet(xq);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return unexpected(Status::faiss_inner_error);
        }
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }

    virtual Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "serilalization on empty index.";
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            // Serialize() is called after Add(), at this time index_ is CPU index actually
            faiss::write_index(index_.get(), &writer);
            std::shared_ptr<uint8_t[]> data(writer.data_);
            binset.Append("FLAT", data, writer.rp);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    virtual Status
    Deserialize(const BinarySet& binset) override {
        auto binary = binset.GetByName("FLAT");
        MemoryIOReader reader;
        try {
            reader.total = binary->size;
            reader.data_ = binary->data.get();
            std::unique_ptr<faiss::Index> index(faiss::read_index(&reader));

            auto gpu_res = GPUResMgr::GetInstance().GetRes();
            ResScope rs(gpu_res, true);
            auto gpu_index = faiss::gpu::index_cpu_to_gpu(gpu_res->faiss_res_.get(), gpu_res->gpu_id_, index.get());
            index_.reset(gpu_index);
            res_ = gpu_res;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error, " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, const LoadConfig& config) override {
        return Status::not_implemented;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<GpuFlatConfig>();
    }

    virtual int64_t
    Dim() const override {
        return index_->d;
    }

    virtual int64_t
    Size() const override {
        return index_->ntotal * index_->d * sizeof(float);
    }

    virtual int64_t
    Count() const override {
        return index_->ntotal;
    }

    virtual std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FAISS_GPU_IDMAP;
    }

 private:
    mutable ResWPtr res_;
    std::unique_ptr<faiss::Index> index_;
};

KNOWHERE_REGISTER_GLOBAL(GPU_FAISS_FLAT, [](const Object& object) { return Index<GpuFlatIndexNode>::Create(object); });

}  // namespace knowhere
