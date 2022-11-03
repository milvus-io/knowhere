#pragma once

#ifndef _WINDOWS

#include <mutex>
#include <queue>
#include <libaio.h>
#include <condition_variable>
#include "utils.h"
#include "concurrent_queue.h"

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

  static void InitGlobalAioPool(size_t num_ctx, size_t max_events);

  static std::shared_ptr<AioContextPool> GetGlobalAioPool();

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

  AioContextPool(size_t num_ctx, size_t max_events);
};

#endif
