#pragma once

#include <algorithm>
#include <thread>
#include <unordered_map>
#include <vector>

namespace hnswlib {

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    int numelements;
    std::unordered_map<std::thread::id, std::vector<bool>> map;

 public:
    VisitedListPool(int numelements1) {
        numelements = numelements1;
    }

    std::vector<bool>&
    getFreeVisitedList() {
        auto& res = map[std::this_thread::get_id()];
        if (res.size() != numelements) {
            res.assign(numelements, false);
        } else {
            std::fill(res.begin(), res.end(), false);
        }
        return res;
    };

    int64_t
    size() {
        return numelements * (sizeof(std::thread::id) + numelements / 8) + sizeof(*this);
    }
};
}  // namespace hnswlib
