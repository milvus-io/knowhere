#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H
#include "knowhere/bitsetview.h"
#include "knowhere/dataset.h"
#include "knowhere/knowhere.h"

namespace knowhere {

class BruteForce {
 public:
    static DataSetPtr
    Search(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
           const knowhere::BitsetView bitset);

    static DataSetPtr
    RangeSearch(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                const knowhere::BitsetView bitset);
};

}  // namespace knowhere

#endif /* BRUTE_FORCE_H */
