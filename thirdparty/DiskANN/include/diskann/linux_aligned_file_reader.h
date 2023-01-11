// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS

#include "aligned_file_reader.h"
#include "aio_context_pool.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t     file_sz;
  FileHandle   file_desc;
  io_context_t bad_ctx = (io_context_t) -1;
  std::shared_ptr<AioContextPool> ctx_pool_;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  io_context_t get_ctx() {
    return ctx_pool_->pop();
  }

  void put_ctx(io_context_t ctx) {
    ctx_pool_->push(ctx);
  }

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false);
};

#endif
