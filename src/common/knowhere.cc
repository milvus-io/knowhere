#include "knowhere/knowhere.h"
namespace knowhere {
Index<IndexNode>
IndexFactory::Create(const std::string& name) {
    auto& func_mapping_ = MapInstance();
    assert(func_mapping_.find(name) != func_mapping_.end());
    return func_mapping_[name]();
}

const IndexFactory&
IndexFactory::Register(const std::string& name, std::function<Index<IndexNode>()> func) {
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
