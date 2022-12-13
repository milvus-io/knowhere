#include "knowhere/log.h"
namespace knowhere {

KnowhereException::KnowhereException(std::string msg) : msg_(std::move(msg)) {
}

KnowhereException::KnowhereException(const std::string& m, const char* funcName, const char* file, int line) {
    std::string filename;
    try {
        size_t pos;
        std::string file_path(file);
        pos = file_path.find_last_of('/');
        filename = file_path.substr(pos + 1);
    } catch (std::exception& e) {
        LOG_KNOWHERE_DEBUG_ << e.what();
    }

    int size = snprintf(nullptr, 0, "Error in %s at %s:%d: %s", funcName, filename.c_str(), line, m.c_str());
    msg_.resize(size + 1);
    snprintf(&msg_[0], msg_.size(), "Error in %s at %s:%d: %s", funcName, filename.c_str(), line, m.c_str());
}

const char*
KnowhereException::what() const noexcept {
    return msg_.c_str();
}

}  // namespace knowhere
