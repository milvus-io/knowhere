#ifndef ANNOY_CONFIG_H
#define ANNOY_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {
class AnnoyConfig : public BaseConfig {
 public:
    int n_trees;
    int search_k;
    KNOHWERE_DECLARE_CONFIG(AnnoyConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(n_trees).description("annoy n_trees.").set_default(8).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_k).description("annoy search k.").set_default(100).for_search();
    }
};

}  // namespace knowhere

#endif /* ANNOY_CONFIG_H */
