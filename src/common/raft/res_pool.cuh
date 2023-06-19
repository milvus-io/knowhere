#include "knowhere/log.h"
#include "raft/core/device_resources.hpp"

namespace raft_res_pool {

struct context {
    context()
        : resources_(
              []() {
                  return new rmm::cuda_stream();  // Avoid program exit datart
                                                  // unload error
              }()
                  ->view(),
              nullptr, rmm::mr::get_current_device_resource()) {
    }
    ~context() = default;
    context(context&&) = delete;
    context(context const&) = delete;
    context&
    operator=(context&&) = delete;
    context&
    operator=(context const&) = delete;
    raft::device_resources resources_;
};

inline context&
get_context() {
    thread_local context ctx;
    return ctx;
};
class resource {
 public:
    static resource&
    instance();
    void
    set_pool_size(std::size_t init_size, std::size_t max_size);

    void
    init(rmm::cuda_device_id device_id);

 private:
    resource(){};
    ~resource(){};
    resource(resource&&) = delete;
    resource(resource const&) = delete;
    resource&
    operator=(resource&&) = delete;
    resource&
    operator=(context const&) = delete;
    rmm::mr::cuda_memory_resource up_mr_;
    std::map<rmm::cuda_device_id::value_type,
             std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>>
        map_;
    mutable std::mutex mtx_;
    std::size_t initial_pool_size = 2048;  // MB
    std::size_t maximum_pool_size = 4096;  // MB
};

};  // namespace raft_res_pool
