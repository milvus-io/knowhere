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

#include <memory>
#include <string>

#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/IndexIVFHNSW.h"
#include "knowhere/index/vector_index/IndexIVFPQ.h"
#include "knowhere/index/vector_index/IndexIVFSQ.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/index/vector_offset_index/IndexIVF_NM.h"

#ifdef KNOWHERE_GPU_VERSION
#include "knowhere/index/vector_index/gpu/IndexGPUIVF.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFPQ.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFSQ.h"
#include "knowhere/index/vector_index/gpu/IndexIVFSQHybrid.h"
#include "knowhere/index/vector_offset_index/gpu/IndexGPUIVF_NM.h"
#endif

constexpr int DEVICEID = 0;
constexpr int64_t DIM = 128;
constexpr int64_t NB = 10000;
constexpr int64_t NQ = 10;
constexpr int64_t K = 10;
constexpr int64_t PINMEM = 1024 * 1024 * 200;
constexpr int64_t TEMPMEM = 1024 * 1024 * 300;
constexpr int64_t RESNUM = 2;

class ParamGenerator {
 public:
    static ParamGenerator&
    GetInstance() {
        static ParamGenerator instance;
        return instance;
    }

    knowhere::Config
    Gen(const knowhere::IndexType& type) {
        if (type == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            return knowhere::Config{
                {knowhere::Meta::METRIC_TYPE, knowhere::MetricEnum::L2},
                {knowhere::Meta::DIM, DIM},
                {knowhere::Meta::TOPK, K},
                {knowhere::Meta::DEVICEID, DEVICEID},
                {knowhere::IndexParams::nlist, 100},
                {knowhere::IndexParams::nprobe, 4},
                {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
            };
        } else if (type == knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
            return knowhere::Config{
                {knowhere::Meta::METRIC_TYPE, knowhere::MetricEnum::L2},
                {knowhere::Meta::DIM, DIM},
                {knowhere::Meta::TOPK, K},
                {knowhere::Meta::DEVICEID, DEVICEID},
                {knowhere::IndexParams::nlist, 100},
                {knowhere::IndexParams::nprobe, 4},
                {knowhere::IndexParams::m, 4},
                {knowhere::IndexParams::nbits, 8},
                {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
            };
        } else if (type == knowhere::IndexEnum::INDEX_FAISS_IVFSQ8 ||
                   type == knowhere::IndexEnum::INDEX_FAISS_IVFSQ8H) {
            return knowhere::Config{
                {knowhere::Meta::METRIC_TYPE, knowhere::MetricEnum::L2},
                {knowhere::Meta::DIM, DIM},
                {knowhere::Meta::TOPK, K},
                {knowhere::Meta::DEVICEID, DEVICEID},
                {knowhere::IndexParams::nlist, 100},
                {knowhere::IndexParams::nprobe, 4},
                {knowhere::IndexParams::nbits, 8},
                {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
            };
        } else if (type == knowhere::IndexEnum::INDEX_FAISS_IVFHNSW) {
            return knowhere::Config{
                {knowhere::Meta::METRIC_TYPE, knowhere::MetricEnum::L2},
                {knowhere::Meta::DIM, DIM},
                {knowhere::Meta::TOPK, K},
                {knowhere::Meta::DEVICEID, DEVICEID},
                {knowhere::IndexParams::nlist, 100},
                {knowhere::IndexParams::nprobe, 4},
                {knowhere::IndexParams::M, 16},
                {knowhere::IndexParams::efConstruction, 200},
                {knowhere::IndexParams::ef, 200},
                {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
            };
        } else {
            std::cout << "Invalid index type " << type << std::endl;
        }
        return knowhere::Config();
    }
};

#include <gtest/gtest.h>

class TestGpuIndexBase : public ::testing::Test {
 protected:
    void
    SetUp() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICEID, PINMEM, TEMPMEM, RESNUM);
#endif
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }
};
