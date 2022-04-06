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

#include <string>

#include "knowhere/common/BinarySet.h"
#include "knowhere/common/Config.h"
#include "knowhere/common/Exception.h"

namespace knowhere {

extern const char* INDEX_FILE_SLICE_SIZE_IN_MEGABYTE;
extern int64_t index_file_slice_size;

void
Assemble(BinarySet& binarySet);

void
Disassemble(BinarySet& binarySet, const Config& config);

void
AppendSliceMeta(BinarySet& binarySet, const Config& meta_info);

BinaryPtr
EraseSliceMeta(BinarySet& binarySet);

}  // namespace knowhere
