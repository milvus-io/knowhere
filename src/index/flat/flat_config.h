#ifndef FLAT_CONFIG_H
#define FLAT_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class FlatConfig : public Config {
 public:
    int dim;
    std::string metric_type;
    int k;
    float radius;
    KNOHWERE_DECLARE_CONFIG(FlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(dim).description("vector dims").for_all();
        KNOWHERE_CONFIG_DECLARE_FIELD(metric_type).set_default("L2").description("distance metric type").for_all();
        KNOWHERE_CONFIG_DECLARE_FIELD(k).set_default(10).description("top k").for_query().for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(radius).set_default(0.0f).description("radius").for_range();
    }
};

}  // namespace knowhere

#endif /* FLAT_CONFIG_H */
