// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once
#include <sys/fcntl.h>
#include <unistd.h>

#include <stdexcept>

namespace knowhere {
struct FileReader {
    int fd;
    size_t size;

    FileReader(const std::string& filename, bool auto_remove = false) {
        fd = open(filename.data(), O_RDONLY);
        if (fd < 0) {
            std::runtime_error("Cannot open file");
        }

        size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);

        if (auto_remove) {
            unlink(filename.data());
        }
    }

    ssize_t
    read(char* dst, size_t n) {
        return ::read(fd, dst, n);
    }

    off_t
    seek(off_t offset) {
        return lseek(fd, offset, SEEK_SET);
    }

    off_t
    advance(off_t offset) {
        return lseek(fd, offset, SEEK_CUR);
    }

    off_t
    offset() {
        return lseek(fd, 0, SEEK_CUR);
    }

    int
    close() {
        return ::close(fd);
    }
};
}  // namespace knowhere
