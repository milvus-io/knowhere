#include "cagra_config.h"
#include "common/raft/res_pool.cuh"
#include "common/raft_metric.h"
#include "knowhere/index_node.h"
#include "knowhere/log.h"
#include "raft/neighbors/cagra.cuh"
#include "raft/neighbors/cagra_serialize.cuh"
namespace knowhere {

using idx_type = uint32_t;

using cagra_index = raft::neighbors::experimental::cagra::index<float, idx_type>;

namespace detail {
struct device_setter {
    device_setter(int new_device)
        : prev_device_{[]() {
              auto result = int{};
              RAFT_CUDA_TRY(cudaGetDevice(&result));
              return result;
          }()} {
        RAFT_CUDA_TRY(cudaSetDevice(new_device));
    }

    ~device_setter() {
        RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
    }

 private:
    int prev_device_;
};
}  // namespace detail
class CagraIndexNode : public IndexNode {
 public:
    CagraIndexNode(const Object& object) : devs_{}, gpu_index_{} {
    }

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success)
            return err;
        return Status::success;
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) override {
        auto cagra_cfg = static_cast<const knowhere::CagraConfig&>(cfg);
        if (gpu_index_) {
            LOG_KNOWHERE_WARNING_ << "index is already trained";
            return Status::index_already_trained;
        }
        if (cagra_cfg.gpu_ids.size() != 1) {
            LOG_KNOWHERE_WARNING_ << "Cagra implementation is single-GPU only" << std::endl;
            return Status::raft_inner_error;
        }
        auto metric = Str2RaftMetricType(cagra_cfg.metric_type);
        if (!metric.has_value()) {
            LOG_KNOWHERE_WARNING_ << "please check metric value: " << cagra_cfg.metric_type;
            return metric.error();
        }
        if (metric.value() != raft::distance::DistanceType::L2Expanded) {
            LOG_KNOWHERE_WARNING_ << "only support L2Expanded metric type";
            return Status::invalid_metric_type;
        }
        devs_.insert(devs_.begin(), cagra_cfg.gpu_ids.begin(), cagra_cfg.gpu_ids.end());
        auto scoped_device = detail::device_setter{*cagra_cfg.gpu_ids.begin()};
        auto build_params = raft::neighbors::experimental::cagra::index_params{};
        build_params.intermediate_graph_degree = cagra_cfg.intermediate_graph_degree;
        build_params.graph_degree = cagra_cfg.graph_degree;
        build_params.metric = metric.value();
        thread_local rmm::cuda_stream stream;
        thread_local rmm::mr::cuda_memory_resource mr;
        thread_local raft::device_resources res(stream.view(), nullptr, &mr);
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
        auto data_gpu = raft::make_device_matrix<float, idx_type>(res, rows, dim);
        RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data_handle(), data, data_gpu.size() * sizeof(float), cudaMemcpyDefault,
                                      stream.value()));
        gpu_index_ = raft::neighbors::experimental::cagra::build(
            res, build_params,
            raft::make_device_matrix_view<const float, idx_type>((const float*)data_gpu.data_handle(), rows, dim));
        this->dim_ = dim;
        this->counts_ = rows;
        res.sync_stream();
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        return Status::success;
    }

    virtual expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto cagra_cfg = static_cast<const CagraConfig&>(cfg);
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
        auto output_size = rows * cagra_cfg.k;
        auto ids = std::unique_ptr<idx_type[]>(new idx_type[output_size]);
        auto dis = std::unique_ptr<float[]>(new float[output_size]);
        try {
            auto scoped_device = detail::device_setter{devs_[0]};
            auto* res_ = &raft_res_pool::get_context().resources_;

            auto data_gpu = raft::make_device_matrix<float, idx_type>(*res_, rows, dim);
            raft::copy(data_gpu.data_handle(), data, data_gpu.size(), res_->get_stream());

            auto search_params = raft::neighbors::experimental::cagra::search_params{};
            search_params.max_queries = cagra_cfg.max_queries;
            auto ids_dev = raft::make_device_matrix<idx_type, idx_type>(*res_, rows, cagra_cfg.k);
            auto dis_dev = raft::make_device_matrix<float, idx_type>(*res_, rows, cagra_cfg.k);
            raft::neighbors::experimental::cagra::search(*res_, search_params, *gpu_index_,
                                                         raft::make_const_mdspan(data_gpu.view()), ids_dev.view(),
                                                         dis_dev.view());

            raft::copy(ids.get(), ids_dev.data_handle(), output_size, res_->get_stream());
            raft::copy(dis.get(), dis_dev.data_handle(), output_size, res_->get_stream());
            res_->sync_stream();

        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return Status::raft_inner_error;
        }
        return Status::not_implemented;
        //        return GenResultDataSet(rows, cagra_cfg.k, ids.release(), dis.release());
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return Status::not_implemented;
    }

    virtual expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
        return Status::not_implemented;
    }

    virtual bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        return Status::not_implemented;
    }

    virtual Status
    Serialize(BinarySet& binset) const override {
        if (!gpu_index_.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty RaftCagraIndex.";
            return Status::empty_index;
        }
        std::stringbuf buf;
        std::ostream os(&buf);
        os.write((char*)(&this->dim_), sizeof(this->dim_));
        os.write((char*)(&this->counts_), sizeof(this->counts_));
        os.write((char*)(&this->devs_[0]), sizeof(this->devs_[0]));

        auto scoped_device = detail::device_setter{devs_[0]};
        rmm::mr::cuda_memory_resource mr;
        rmm::cuda_stream stm;
        raft::device_resources res(stm.view(), nullptr, &mr);

        raft::neighbors::experimental::cagra::serialize<float, idx_type>(res, os, *gpu_index_);

        os.flush();
        std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);

        memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
        binset.Append(this->Type(), index_binary, buf.str().size());
        return Status::success;
    }

    virtual Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        std::stringbuf buf;
        auto binary = binset.GetByName(this->Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }
        buf.sputn((char*)binary->data.get(), binary->size);
        std::istream is(&buf);

        is.read((char*)(&this->dim_), sizeof(this->dim_));
        is.read((char*)(&this->counts_), sizeof(this->counts_));
        this->devs_.resize(1);
        is.read((char*)(&this->devs_[0]), sizeof(this->devs_[0]));
        auto scoped_device = detail::device_setter{devs_[0]};

        thread_local rmm::mr::cuda_memory_resource mr;
        thread_local rmm::cuda_stream stream;
        thread_local raft::device_resources res(stream.view(), nullptr, &mr);

        cagra_index index_ = raft::neighbors::experimental::cagra::deserialize<float, idx_type>(res, is);
        is.sync();
        gpu_index_ = cagra_index(std::move(index_));

        return Status::success;
    }

    virtual Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<CagraConfig>();
    }

    virtual int64_t
    Dim() const override {
        return dim_;
    }

    virtual int64_t
    Size() const override {
        return 0;
    }

    virtual int64_t
    Count() const override {
        return counts_;
    }

    virtual std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_RAFT_IVFFLAT;
    }

 private:
    std::vector<int32_t> devs_;
    int64_t dim_ = 0;
    int64_t counts_ = 0;
    std::optional<cagra_index> gpu_index_;
};

}  // namespace knowhere
