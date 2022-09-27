#ifndef FLAT_GPU_CONFIG_H
#define FLAT_GPU_CONFIG_H

#include "index/flat/flat_config.h"

namespace knowhere {

class GpuFlatConfig : public FlatConfig {
 public:
    CFG_LIST gpu_ids;
    KNOHWERE_DECLARE_CONFIG(GpuFlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("the gpu id, which device use")
            .set_default({
                0,
            })
            .for_train();
    }
};

}  // namespace knowhere

#endif /* FLAT_GPU_CONFIG_H */
