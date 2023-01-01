#ifndef IVF_CONFIG_H
#define IVF_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class IvfConfig : public BaseConfig {
 public:
    int nlist;
    int nprobe;
    KNOHWERE_DECLARE_CONFIG(IvfConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(nlist).set_default(1024).description("number of inverted lists.").for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(nprobe)
            .set_default(1024)
            .description("number of probes at query time.")
            .for_search()
            .for_range_search();
    }
};

class IvfFlatConfig : public IvfConfig {};

class IvfPqConfig : public IvfConfig {
 public:
    int m;
    int nbits;
    KNOHWERE_DECLARE_CONFIG(IvfPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(m).description("m").set_default(4).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(nbits).description("nbits").set_default(8).for_train();
    }
};

class IvfSqConfig : public IvfConfig {};

class IvfBinConfig : public IvfConfig {};

}  // namespace knowhere

#endif /* IVF_CONFIG_H */
