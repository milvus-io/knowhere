#include "res_pool.cuh"
namespace raft_res_pool {

resource&
resource::instance() {
    static resource res;
    return res;
}

void
resource::set_pool_size(std::size_t init_size, std::size_t max_size) {
    this->initial_pool_size = init_size;
    this->maximum_pool_size = max_size;
}

void
resource::init(rmm::cuda_device_id device_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = map_.find(device_id.value());
    if (it == map_.end()) {
        char* env_str = getenv("KNOWHERE_GPU_MEM_POOL_SIZE");
        if (env_str != NULL) {
            std::size_t initial_pool_size_tmp, maximum_pool_size_tmp;
            auto stat = sscanf(env_str, "%zu;%zu", &initial_pool_size_tmp, &maximum_pool_size_tmp);
            if (stat == 2) {
                LOG_KNOWHERE_INFO_ << "Get Gpu Pool Size From env, init size: " << initial_pool_size_tmp
                                   << " MB, max size: " << maximum_pool_size_tmp << " MB";
                this->initial_pool_size = initial_pool_size_tmp;
                this->maximum_pool_size = maximum_pool_size_tmp;
            } else {
                LOG_KNOWHERE_WARNING_ << "please check env format";
            }
        }

        auto mr_ = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            &up_mr_, initial_pool_size << 20, maximum_pool_size << 20);
        rmm::mr::set_per_device_resource(device_id, mr_.get());
        map_[device_id.value()] = std::move(mr_);
    }
}

};  // namespace raft_res_pool
