#ifndef HNSW_CONFIG_H
#define HNSW_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {
class HnswConfig : public BaseConfig {
 public:
    int M;
    int efConstruction;
    int ef;
    KNOHWERE_DECLARE_CONFIG(HnswConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(M).description("hnsw M").set_default(16).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(efConstruction).description("hnsw efConstruction").set_default(200).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef).description("hnsw ef").set_default(32).for_search();
    }
};

}  // namespace knowhere

#endif /* HNSW_CONFIG_H */
