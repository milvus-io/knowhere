// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct FileIOWriter {
    std::fstream fs;
    std::string name;

    explicit FileIOWriter(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::out | std::ios::binary);
    }

    ~FileIOWriter() {
        fs.close();
    }

    size_t operator()(void* ptr, size_t size) {
        fs.write(reinterpret_cast<char*>(ptr), size);
        return size;
    }
};

struct FileIOReader {
    std::fstream fs;
    std::string name;

    explicit FileIOReader(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::in | std::ios::binary);
    }

    ~FileIOReader() {
        fs.close();
    }

    size_t operator()(void* ptr, size_t size) {
        fs.read(reinterpret_cast<char*>(ptr), size);
        return size;
    }

    size_t size() {
        fs.seekg(0, fs.end);
        size_t len = fs.tellg();
        fs.seekg(0, fs.beg);
        return len;
    }
};
