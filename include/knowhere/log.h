#ifndef KNOWHERE_LOG_H
#define KNOWHERE_LOG_H

#include "easylogging++.h"
#define KNOWHERE_MODULE_NAME "KNOWHERE"
#define KNOWHERE_MODULE_CLASS_FUNCTION                                                                \
    knowhere::LogOut("[%s][%s::%s][%s] ", KNOWHERE_MODULE_NAME, (typeid(*this).name()), __FUNCTION__, \
                     knowhere::GetThreadName().c_str())
#define KNOWHERE_MODULE_FUNCTION \
    knowhere::LogOut("[%s][%s][%s] ", KNOWHERE_MODULE_NAME, __FUNCTION__, knowhere::GetThreadName().c_str())

#define LOG_KNOWHERE_TRACE_ LOG(TRACE) << KNOWHERE_MODULE_FUNCTION
#define LOG_KNOWHERE_DEBUG_ LOG(DEBUG) << KNOWHERE_MODULE_FUNCTION
#define LOG_KNOWHERE_INFO_ LOG(INFO) << KNOWHERE_MODULE_FUNCTION
#define LOG_KNOWHERE_WARNING_ LOG(WARNING) << KNOWHERE_MODULE_FUNCTION
#define LOG_KNOWHERE_ERROR_ LOG(ERROR) << KNOWHERE_MODULE_FUNCTION
#define LOG_KNOWHERE_FATAL_ LOG(FATAL) << KNOWHERE_MODULE_FUNCTION

namespace knowhere {
class KnowhereException : public std::exception {
 public:
    explicit KnowhereException(std::string msg);

    KnowhereException(const std::string& msg, const char* funName, const char* file, int line);

    const char*
    what() const noexcept override;

    std::string msg_;
};

inline std::string
LogOut(const char* pattern, ...) {
    size_t len = strnlen(pattern, 1024) + 256;
    auto str_p = std::make_unique<char[]>(len);
    memset(str_p.get(), 0, len);

    va_list vl;
    va_start(vl, pattern);
    vsnprintf(str_p.get(), len - 1, pattern, vl);  // NOLINT
    va_end(vl);

    return std::string(str_p.get());
}

inline void
SetThreadName(const std::string& name) {
#ifdef __APPLE__
    pthread_setname_np(name.c_str());
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name.c_str());
#endif
}

inline std::string
GetThreadName() {
    std::string thread_name = "unamed";
    char name[16];
    size_t len = 16;
    auto err = pthread_getname_np(pthread_self(), name, len);
    if (not err) {
        thread_name = name;
    }

    return thread_name;
}

inline void
log_trace_(const std::string& s) {
    LOG_KNOWHERE_TRACE_ << s;
}

inline void
log_debug_(const std::string& s) {
    LOG_KNOWHERE_DEBUG_ << s;
}

inline void
log_info_(const std::string& s) {
    LOG_KNOWHERE_INFO_ << s;
}

inline void
log_warning_(const std::string& s) {
    LOG_KNOWHERE_WARNING_ << s;
}

inline void
log_error_(const std::string& s) {
    LOG_KNOWHERE_ERROR_ << s;
}

inline void
log_fatal_(const std::string& s) {
    LOG_KNOWHERE_FATAL_ << s;
}

/*
 * Please use LOG_MODULE_LEVEL_C macro in member function of class
 * and LOG_MODULE_LEVEL_ macro in other functions.
 */

/////////////////////////////////////////////////////////////////////////////////////////////////
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

}  // namespace knowhere
#endif /* KNOWHERE_LOG_H */
