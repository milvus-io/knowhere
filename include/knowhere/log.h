#ifndef KNOWHERE_LOG_H
#define KNOWHERE_LOG_H

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"

#define BACKTRACE_SIZE 16

#define KNOWHERE_TO_STRING(x) #x

#if defined(CONSOLE_LOGGING)
#define LOG_KEY_NAME KNOWHERE_TO_STRING(console)
#else
#define LOG_KEY_NAME KNOWHERE_TO_STRING(filelog)
#endif

#define KNOWHERE_INFO(...) SPDLOG_LOGGER_INFO(spdlog::get(LOG_KEY_NAME), __VA_ARGS__)

#define KNOWHERE_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::get(LOG_KEY_NAME), __VA_ARGS__)

#define KNOWHERE_WARN(...) SPDLOG_LOGGER_WARN(spdlog::get(LOG_KEY_NAME), __VA_ARGS__)

#define KNOWHERE_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::get(LOG_KEY_NAME), __VA_ARGS__)

#define KNOWHERE_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::get(LOG_KEY_NAME), __VA_ARGS__)

#define KNOWHERE_DUMP_BACKTRACE                                    \
    {                                                              \
        spdlog::get(LOG_KEY_NAME)->set_level(spdlog::level::info); \
        spdlog::get(LOG_KEY_NAME)->dump_backtrace();               \
    }

#endif /* LOG_H */
