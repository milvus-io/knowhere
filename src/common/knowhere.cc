#include "knowhere/knowhere.h"
namespace knowhere {

Index<IndexNode>
IndexFactory::Create(const std::string& name, const Object& object) {
    auto& func_mapping_ = MapInstance();
    assert(func_mapping_.find(name) != func_mapping_.end());
    LOG_KNOWHERE_INFO_ << "create knowhere index " << name;
    return func_mapping_[name](object);
}

const IndexFactory&
IndexFactory::Register(const std::string& name, std::function<Index<IndexNode>(const Object&)> func) {
    auto& func_mapping_ = MapInstance();
    func_mapping_[name] = func;
    return *this;
}

IndexFactory&
IndexFactory::Instance() {
    static IndexFactory factory;
    return factory;
}

IndexFactory::IndexFactory() {
}
IndexFactory::FuncMap&
IndexFactory::MapInstance() {
    static FuncMap func_map;
    return func_map;
}

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
