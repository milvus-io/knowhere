#ifndef SIMPLE_CONFIG_H
#define SIMPLE_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class SimpleConfig : public BaseConfig {
 public:
    KNOHWERE_DECLARE_CONFIG(SimpleConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(k).set_default(15).for_all();
    }
};

}  // namespace knowhere

#endif /* SIMPLE_CONFIG_H */
