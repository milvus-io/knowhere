// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>
#include "tsl/robin_map.h"
#include "utils.h"

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  void execute_io(io_context_t ctx, uint64_t maxnr, int fd,
                  const std::vector<AlignedRead> &read_reqs,
                  uint64_t n_retries = 0) {
#ifdef DEBUG
    for (auto &req : read_reqs) {
      assert(IS_ALIGNED(req.len, 512));
      // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
      assert(IS_ALIGNED(req.offset, 512));
      assert(IS_ALIGNED(req.buf, 512));
      // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    // break-up requests into chunks of size maxnr each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), maxnr) / maxnr;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops = std::min((uint64_t) read_reqs.size() - (iter * maxnr),
                                (uint64_t) maxnr);
      std::vector<iocb_t *>    cbs(n_ops, nullptr);
      std::vector<io_event_t>  evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * maxnr].buf,
                      read_reqs[j + iter * maxnr].len,
                      read_reqs[j + iter * maxnr].offset);
      }

      // initialize `cbs` using `cb` array
      //

      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          std::stringstream err;
          err << "io_submit() failed; returned " << ret
              << ", expected=" << n_ops << ", ernno=" << errno << "="
              << ::strerror(-ret) << ", try #" << n_tries + 1
              << ", ctx: " << ctx;
          throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        } else {
          // wait on io_getevents
          ret = io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(),
                             nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            std::stringstream err;
            err << "io_getevents() failed; returned " << ret
                << ", expected=" << n_ops << ", ernno=" << errno << "="
                << ::strerror(-ret) << ", try #" << n_tries + 1;
            throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
          } else {
            break;
          }
        }
      }
      // disabled since req.buf could be an offset into another buf
      /*
      for (auto &req : read_reqs) {
        // corruption check
        assert(malloc_usable_size(req.buf) >= req.len);
      }
      */
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
  this->ctx_pool_ = AioContextPool::GetGlobalAioPool();
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                  << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

void LinuxAlignedFileReader::open(const std::string &fname) {
  int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
  this->file_desc = ::open(fname.c_str(), flags);
  // error checks
  assert(this->file_desc != -1);
  LOG(DEBUG) << "Opened file : " << fname;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                  IOContext& ctx, bool async) {
  if (async == true) {
    diskann::cout << "Async currently not supported in linux." << std::endl;
  }
  assert(this->file_desc != -1);
  //#pragma omp critical
  //	std::cout << "thread: " << std::this_thread::get_id() << ", crtx: " <<
  // ctx
  //<< "\n";
  execute_io(ctx, this->ctx_pool_->max_events_per_ctx(), this->file_desc,
             read_reqs);
}
