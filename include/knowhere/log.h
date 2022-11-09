#ifndef KNOWHERE_LOG_H
#define KNOWHERE_LOG_H

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"

namespace knowhere {
class KnowhereException : public std::exception {
 public:
    explicit KnowhereException(std::string msg);

    KnowhereException(const std::string& msg, const char* funName, const char* file, int line);

    const char*
    what() const noexcept override;

    std::string msg_;
};
}  // namespace knowhere

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

#define KNOWHERE_CHECK(x) \
    if (!(x))             \
        KNOWHERE_ERROR("check failed, " #x);

#define KNOWHERE_THROW_MSG(MSG)                                                \
    do {                                                                       \
        throw KnowhereException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

#define KNOWHERE_THROW_FORMAT(FMT, ...)                                        \
    do {                                                                       \
        std::string __s;                                                       \
        int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__);                   \
        __s.resize(__size + 1);                                                \
        snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__);                       \
        throw KnowhereException(__s, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

#define KNOWHERE_THROW_FMT KNOWHERE_THROW_FORMAT

#define KNOWHERE_THROW_IF_NOT(X)                          \
    do {                                                  \
        if (!(X)) {                                       \
            KNOWHERE_THROW_FMT("Error: '%s' failed", #X); \
        }                                                 \
    } while (false)

#define KNOWHERE_THROW_IF_NOT_MSG(X, MSG)                       \
    do {                                                        \
        if (!(X)) {                                             \
            KNOWHERE_THROW_FMT("Error: '%s' failed: " MSG, #X); \
        }                                                       \
    } while (false)

#define KNOWHERE_THROW_IF_NOT_FMT(X, FMT, ...)                               \
    do {                                                                     \
        if (!(X)) {                                                          \
            KNOWHERE_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
        }                                                                    \
    } while (false)

#endif /* KNOWHERE_LOG_H */
