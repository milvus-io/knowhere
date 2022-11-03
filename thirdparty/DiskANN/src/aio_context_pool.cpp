#ifndef _WINDOWS

#include "aio_context_pool.h"
#include <mutex>

namespace {
  size_t           global_aio_pool_size = 0;
  size_t           global_aio_max_events = 0;
  std::mutex       global_aio_pool_mut;
  const size_t     default_pool_size = std::thread::hardware_concurrency() * 100;
  constexpr size_t default_max_events = 32;
}  // namespace

AioContextPool::AioContextPool(size_t num_ctx, size_t max_events)
    : num_ctx_(num_ctx), max_events_(max_events) {
  for (size_t i = 0; i < num_ctx_; ++i) {
    io_context_t ctx = 0;
    int          ret = io_setup(max_events, &ctx);

    if (ret != 0) {
      assert(-ret != EAGAIN);
      assert(-ret != ENOMEM);
      LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << -ret
                 << ":" << ::strerror(-ret);
    } else {
      LOG(DEBUG) << "allocating ctx: " << ctx;
      ctx_q_.push(ctx);
      ctx_bak_.push_back(ctx);
    }
  }
}

void AioContextPool::InitGlobalAioPool(size_t num_ctx, size_t max_events) {
  if (num_ctx <= 0) {
    LOG(ERROR) << "num_ctx should be bigger than 0";
    return;
  }
  if (global_aio_pool_size == 0) {
    std::scoped_lock lk(global_aio_pool_mut);
    if (global_aio_pool_size == 0) {
      global_aio_pool_size = num_ctx;
      global_aio_max_events = max_events;
      return;
    }
  }
  LOG(WARNING)
      << "Global AioContextPool has already been inialized with context num: "
      << global_aio_pool_size;
}

std::shared_ptr<AioContextPool> AioContextPool::GetGlobalAioPool() {
  if (global_aio_pool_size == 0) {
    std::scoped_lock lk(global_aio_pool_mut);
    if (global_aio_pool_size == 0) {
      global_aio_pool_size = default_pool_size;
      global_aio_max_events = default_max_events;
      LOG(WARNING) << "Global AioContextPool has not been inialized yet, init "
                      "it now with context num: "
                   << global_aio_pool_size;
    }
  }
  static auto pool = std::shared_ptr<AioContextPool>(
      new AioContextPool(global_aio_pool_size, global_aio_max_events));
  return pool;
}

#endif