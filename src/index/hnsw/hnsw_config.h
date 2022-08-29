#ifndef HNSW_CONFIG_H
#define HNSW_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {
class HnswConfig : public BaseConfig {
 public:
    int M;
    int efConstruction;
    int build_thread_num;
    int query_thread_num;
    int ef;
    int range_k;
    KNOHWERE_DECLARE_CONFIG(HnswConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(M).description("hnsw M").set_default(16).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(efConstruction).description("hnsw efConstruction").set_default(200).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(build_thread_num)
            .description("hnsw build thread num")
            .set_default(-1)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(query_thread_num)
            .description("hnsw query thread num")
            .set_default(-1)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef).description("hnsw ef").set_default(32).for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(range_k).description("hnsw range k").set_default(20).for_range();
    }
};

}  // namespace knowhere

#endif /* HNSW_CONFIG_H */
