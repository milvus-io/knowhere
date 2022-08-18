#include <iostream>
#include <string>
namespace knowhere {

template <typename... Args>
class ErrorStat {
 public:
    ErrorStat(int code, char* msg) : code(code), msg(msg) {
    }

 private:
    int code;
    std::string_view msg;
};

template <typename T, typename... Args>
class ErrorStat {
 public:
    ErrorStat(int code, char* msg, Args... args) : code(code), msg(msg), result(std::forward<Args>(args)...) {
    }
    ErrorStat(ErrorStat&) = delete;
    ErrorStat(ErrorStat&&) = delete;
    bool
    isSuccess() {
        return code == 0;
    }
    T*
    GetRes() {
        return &result;
    }
    operator std::string() {
        return std::string(msg);
    }

 private:
    int code;
    std::string_view msg;
    T result;
};

}  // namespace knowhere
