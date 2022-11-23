#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace hnswlib {

struct Neighbor {
    unsigned id;
    float distance;
    bool checked;

    Neighbor(unsigned id = -1, float distance = std::numeric_limits<float>::infinity(), bool checked = false)
        : id(id), distance(distance), checked(checked) {
    }

    inline bool
    operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

// Invariant: after every `insert` and `pop`, `cur_` points to
//            the first Neighbor which is unchecked.
class NeighborSet {
 public:
    explicit NeighborSet(size_t capacity = 0) : size_(0), capacity_(capacity), data_(capacity_ + 1) {
    }

    void
    insert(const Neighbor& nbr) {
        if (size_ == capacity_ && nbr.distance >= data_[size_ - 1].distance) {
            return;
        }
        size_t p = std::lower_bound(data_.begin(), data_.begin() + size_, nbr) - data_.begin();
        std::memmove(&data_[p + 1], &data_[p], (size_ - p) * sizeof(Neighbor));
        data_[p] = nbr;
        if (size_ < capacity_) {
            size_++;
        }
        if (p < cur_) {
            cur_ = p;
        }
    }

    Neighbor
    pop() {
        data_[cur_].checked = true;
        size_t pre = cur_;
        while (cur_ < size_ && data_[cur_].checked) {
            cur_++;
        }
        return data_[pre];
    }

    bool
    has_next() const {
        return cur_ < size_;
    }

    size_t
    size() const {
        return size_;
    }
    size_t
    capacity() const {
        return capacity_;
    }

    Neighbor&
    operator[](size_t i) {
        return data_[i];
    }

    const Neighbor&
    operator[](size_t i) const {
        return data_[i];
    }

    void
    clear() {
        size_ = 0;
        cur_ = 0;
    }

 private:
    size_t size_, capacity_, cur_;
    std::vector<Neighbor> data_;
};

class NeighborSetPool {
 public:
    NeighborSet&
    getFreeNeighborSet(size_t capacity) {
        std::unique_lock lk(mtx);
        auto& ret = map[std::this_thread::get_id()];
        lk.unlock();
        if (ret.capacity() != capacity) {
            ret = NeighborSet(capacity);
        } else {
            ret.clear();
        }
        return ret;
    }

    size_t
    size() {
        size_t sz = sizeof(*this) + sizeof(map);
        for (auto& [k, v] : map) {
            sz += sizeof(k) + sizeof(v);
            sz += (v.capacity() + 1) * sizeof(Neighbor);
        }
        return sz;
    }

 private:
    std::unordered_map<std::thread::id, NeighborSet> map;
    std::mutex mtx;
};

}  // namespace hnswlib