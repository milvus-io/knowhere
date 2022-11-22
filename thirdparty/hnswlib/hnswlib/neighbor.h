#pragma once

#include <algorithm>
#include <cstring>

namespace hnswlib {

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool flag) : id{id}, distance{distance}, flag(flag) {
    }

    inline bool
    operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

static inline int
InsertIntoPool(Neighbor* addr, int size, Neighbor nn) {
    int p = std::lower_bound(addr, addr + size, nn) - addr;
    std::memmove(addr + p + 1, addr + p, (size - p) * sizeof(Neighbor));
    addr[p] = nn;
    return p;
}

}  // namespace hnswlib