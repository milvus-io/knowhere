#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H
#include "knowhere/bitsetview.h"
#include "knowhere/dataset.h"
#include "knowhere/factory.h"

namespace knowhere {

class BruteForce {
 public:
    static expected<DataSetPtr, Status>
    Search(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config, const BitsetView& bitset);

    static expected<DataSetPtr, Status>
    RangeSearch(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                const BitsetView& bitset);
    static Status
    SearchWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, int64_t* ids, float* dis,
                  const Json& config, const BitsetView& bitset);
};

}  // namespace knowhere

#endif /* BRUTE_FORCE_H */
