#include "index/ivf/ivf_config.h"

namespace knowhere {

class IvfGpuFlatConfig : public IvfFlatConfig {
 public:
    CFG_LIST gpu_ids;
    KNOHWERE_DECLARE_CONFIG(IvfGpuFlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("the gpu id, which device use")
            .set_default({
                0,
            })
            .for_train();
    }
};

class IvfGpuPqConfig : public IvfPqConfig {
 public:
    CFG_LIST gpu_ids;
    KNOHWERE_DECLARE_CONFIG(IvfGpuPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("the gpu id, which device use")
            .set_default({
                0,
            })
            .for_train();
    }
};

class IvfGpuSqConfig : public IvfSqConfig {
 public:
    CFG_LIST gpu_ids;
    KNOHWERE_DECLARE_CONFIG(IvfGpuSqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("the gpu id, which device use")
            .set_default({
                0,
            })
            .for_train();
    }
};

}  // namespace knowhere
