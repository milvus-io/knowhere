#include "index/ivf/ivf_config.h"

namespace knowhere {

class IvfGpuFlatConfig : public IvfFlatConfig {
 public:
    int gpu_id;
    KNOHWERE_DECLARE_CONFIG(IvfGpuFlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_id).description("the gpu id, which device use").set_default(0).for_all();
    }
};

class IvfGpuPqConfig : public IvfPqConfig {
 public:
    int gpu_id;
    KNOHWERE_DECLARE_CONFIG(IvfGpuPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_id).description("the gpu id, which device use").set_default(0).for_all();
    }
};

class IvfGpuSqConfig : public IvfSqConfig {
 public:
    int gpu_id;
    KNOHWERE_DECLARE_CONFIG(IvfGpuSqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_id).description("the gpu id, which device use").set_default(0).for_all();
    }
};

}  // namespace knowhere
