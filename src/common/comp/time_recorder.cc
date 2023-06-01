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

#include "knowhere/comp/time_recorder.h"

#include <iostream>
#include <utility>

#include "knowhere/log.h"

namespace knowhere {

TimeRecorder::TimeRecorder(std::string hdr, int64_t log_level) : header_(std::move(hdr)), log_level_(log_level) {
    start_ = last_ = stdclock::now();
}

std::string
TimeRecorder::GetTimeSpanStr(double span) {
    std::string str_ms = std::to_string(span * 0.001) + " ms";
    return str_ms;
}

void
TimeRecorder::PrintTimeRecord(const std::string& msg, double span) {
    // for TRACE / DEBUG level, print nothing
    if (log_level_ < 2) {
        return;
    }

    std::string str_log;
    if (!header_.empty()) {
        str_log += header_ + ": ";
    }
    str_log += msg;
    str_log += " (";
    str_log += TimeRecorder::GetTimeSpanStr(span);
    str_log += ")";

    switch (log_level_) {
        case 2:
            LOG_KNOWHERE_INFO_ << str_log;
            break;
        case 3:
            LOG_KNOWHERE_WARNING_ << str_log;
            break;
        case 4:
            LOG_KNOWHERE_ERROR_ << str_log;
            break;
        case 5:
            LOG_KNOWHERE_FATAL_ << str_log;
            break;
        default:
            LOG_KNOWHERE_INFO_ << str_log;
            break;
    }
}

double
TimeRecorder::RecordSection(const std::string& msg) {
    stdclock::time_point curr = stdclock::now();
    double span = (std::chrono::duration<double, std::micro>(curr - last_)).count();
    last_ = curr;

    PrintTimeRecord(msg, span);
    return span;
}

double
TimeRecorder::ElapseFromBegin(const std::string& msg) {
    stdclock::time_point curr = stdclock::now();
    double span = (std::chrono::duration<double, std::micro>(curr - start_)).count();

    PrintTimeRecord(msg, span);
    return span;
}

}  // namespace knowhere
