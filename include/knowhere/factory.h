#ifndef INDEX_FACTORY_H
#define INDEX_FACTORY_H

#include <functional>
#include <string>
#include <unordered_map>

#include "knowhere/index.h"

namespace knowhere {
class IndexFactory {
 public:
    Index<IndexNode>
    Create(const std::string& name, const Object& object = nullptr);
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>(const Object&)> func);
    static IndexFactory&
    Instance();

 private:
    typedef std::map<std::string, std::function<Index<IndexNode>(const Object&)>> FuncMap;
    IndexFactory();
    static FuncMap&
    MapInstance();
};

#define KNOWHERE_CONCAT(x, y) x##y
#define KNOWHERE_REGISTER_GLOBAL(name, func) \
    const IndexFactory& KNOWHERE_CONCAT(index_factory_ref_, name) = IndexFactory::Instance().Register(#name, func)
}  // namespace knowhere

#endif /* INDEX_FACTORY_H */
