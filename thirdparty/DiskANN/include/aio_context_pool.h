#pragma once

#include "knowhere/common/Log.h"
#ifndef _WINDOWS

#include <mutex>
#include <queue>
#include <libaio.h>
#include <condition_variable>
#include "utils.h"
#include "concurrent_queue.h"

namespace {
  constexpr size_t default_max_nr = 65536;
  constexpr size_t default_max_events = 256; // sync this with diskann MAX_N_SECTOR_READS
  const size_t     default_pool_size =
      std::min(size_t(std::thread::hardware_concurrency() * 10),
               default_max_nr / default_max_events);
}  // namespace

class AioContextPool {
 public:
  AioContextPool(const AioContextPool&) = delete;

  AioContextPool& operator=(const AioContextPool&) = delete;

  AioContextPool(AioContextPool&&) noexcept = delete;

  AioContextPool& operator==(AioContextPool&&) noexcept = delete;

  size_t max_events_per_ctx() {
    return max_events_;
  }

  void push(io_context_t ctx) {
    {
      std::scoped_lock lk(ctx_mtx_);
      ctx_q_.push(ctx);
    }
    ctx_cv_.notify_one();
  }

  io_context_t pop() {
    std::unique_lock lk(ctx_mtx_);
    if (stop_) {
      return nullptr;
    }
    ctx_cv_.wait(lk, [this] { return ctx_q_.size(); });
    if (stop_) {
      return nullptr;
    }
    auto ret = ctx_q_.front();
    ctx_q_.pop();
    return ret;
  }

  static void InitGlobalAioPool(size_t num_ctx, size_t max_events) {
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

  static std::shared_ptr<AioContextPool> GetGlobalAioPool() {
    if (global_aio_pool_size == 0) {
      std::scoped_lock lk(global_aio_pool_mut);
      if (global_aio_pool_size == 0) {
        global_aio_pool_size = default_pool_size;
        global_aio_max_events = default_max_events;
        LOG(WARNING)
            << "Global AioContextPool has not been inialized yet, init "
               "it now with context num: "
            << global_aio_pool_size;
      }
    }
    static auto pool = std::shared_ptr<AioContextPool>(
        new AioContextPool(global_aio_pool_size, global_aio_max_events));
    return pool;
  }

  ~AioContextPool() {
    stop_ = true;
    for (auto ctx : ctx_bak_) {
      io_destroy(ctx);
    }
    ctx_cv_.notify_all();
  }

 private:
  std::vector<io_context_t> ctx_bak_;
  std::queue<io_context_t>  ctx_q_;
  std::mutex                ctx_mtx_;
  std::condition_variable   ctx_cv_;
  bool                      stop_ = false;
  size_t                    num_ctx_;
  size_t                    max_events_;
  inline static size_t           global_aio_pool_size = 0;
  inline static size_t           global_aio_max_events = 0;
  inline static std::mutex       global_aio_pool_mut;

  AioContextPool(size_t num_ctx, size_t max_events)
      : num_ctx_(num_ctx), max_events_(max_events) {
    for (size_t i = 0; i < num_ctx_; ++i) {
      io_context_t ctx = 0;
      int          ret = -1;
      for (int retry = 0; (ret = io_setup(max_events, &ctx)) != 0 && retry < 5;
           ++retry) {
        if (-ret != EAGAIN) {
          LOG(ERROR) << "Unknown error occur in io_setup, errno: " << -ret
                     << ", " << strerror(-ret);
        }
      }
      if (ret != 0) {
        LOG(ERROR) << "io_setup() failed; returned " << ret
                   << ", errno=" << -ret << ":" << ::strerror(-ret);
      } else {
        LOG_KNOWHERE_DEBUG_ << "allocating ctx: " << ctx;
        ctx_q_.push(ctx);
        ctx_bak_.push_back(ctx);
      }
    }
  }
};

#endif

