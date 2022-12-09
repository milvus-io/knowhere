#include "knowhere/factory.h"

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
}  // namespace knowhere
