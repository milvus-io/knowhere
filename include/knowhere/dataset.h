#ifndef DATASET_H
#define DATASET_H

#include <any>
#include <map>
#include <memory>
#include <shared_mutex>
#include <utility>
#include <variant>
namespace knowhere {

namespace meta {
const std::string IDS = "ids";
const std::string DISTANCE = "distance";
const std::string LIMS = "lims";
const std::string TENSOR = "tensor";
const std::string ROWS = "rows";
const std::string DIM = "dim";
}  // namespace meta

class DataSet {
 public:
    typedef std::variant<const float*, const size_t*, const int64_t*, const void*, int64_t> Var;
    DataSet() = default;
    ~DataSet() {
        if (!is_owner)
            return;
        for (auto&& x : this->data_) {
            {
                auto ptr = std::get_if<0>(&x.second);
                if (ptr != nullptr)
                    delete[] * ptr;
            }
            {
                auto ptr = std::get_if<1>(&x.second);
                if (ptr != nullptr)
                    delete[] * ptr;
            }
            {
                auto ptr = std::get_if<2>(&x.second);
                if (ptr != nullptr)
                    delete[] * ptr;
            }
            {
                auto ptr = std::get_if<3>(&x.second);
                if (ptr != nullptr)
                    delete[](char*)(*ptr);
            }
        }
    }

    void
    SetDistance(const float* dis) {
        std::unique_lock lock(mutex_);
        this->data_[meta::DISTANCE] = Var(std::in_place_index<0>, dis);
    }

    void
    SetLims(const size_t* lims) {
        std::unique_lock lock(mutex_);
        this->data_[meta::LIMS] = Var(std::in_place_index<1>, lims);
    }

    void
    SetIds(const int64_t* ids) {
        std::unique_lock lock(mutex_);
        this->data_[meta::IDS] = Var(std::in_place_index<2>, ids);
    }

    void
    SetTensor(const void* tensor) {
        std::unique_lock lock(mutex_);
        this->data_[meta::TENSOR] = Var(std::in_place_index<3>, tensor);
    }
    void
    SetRows(const int64_t rows) {
        std::unique_lock lock(mutex_);
        this->data_[meta::ROWS] = Var(std::in_place_index<4>, rows);
    }
    void
    SetDim(const int64_t dim) {
        std::unique_lock lock(mutex_);
        this->data_[meta::DIM] = Var(std::in_place_index<4>, dim);
    }

    const float*
    GetDistance() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::DISTANCE);
        if (it != this->data_.end()) {
            const float* res = *std::get_if<0>(&it->second);
            return res;
        }
        return nullptr;
    }

    const size_t*
    GetLims() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::LIMS);
        if (it != this->data_.end()) {
            const size_t* res = *std::get_if<1>(&it->second);
            return res;
        }
        return nullptr;
    }

    const int64_t*
    GetIds() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::IDS);
        if (it != this->data_.end()) {
            const int64_t* res = *std::get_if<2>(&it->second);
            return res;
        }
        return nullptr;
    }

    const void*
    GetTensor() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::TENSOR);
        if (it != this->data_.end()) {
            const void* res = *std::get_if<3>(&it->second);
            return res;
        }
        return nullptr;
    }
    int64_t
    GetRows() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::ROWS);
        if (it != this->data_.end()) {
            int64_t res = *std::get_if<4>(&it->second);
            return res;
        }
        return 0;
    }

    int64_t
    GetDim() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::DIM);
        if (it != this->data_.end()) {
            int64_t res = *std::get_if<4>(&it->second);
            return res;
        }
        return 0;
    }

    void
    SetIsOwner(bool is_owner) {
        std::unique_lock lock(mutex_);
        this->is_owner = is_owner;
    }

 private:
    mutable std::shared_mutex mutex_;
    std::map<std::string, Var> data_;
    bool is_owner = true;
};
using DataSetPtr = std::shared_ptr<DataSet>;
}  // namespace knowhere
#endif /* DATASET_H */
