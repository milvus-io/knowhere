#ifndef INDEX_NODE_H
#define INDEX_NODE_H

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/object.h"

namespace knowhere {

class IndexNode : public Object {
 public:
    virtual Status
    Build(const DataSet& dataset, const Config& cfg) = 0;

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) = 0;

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) = 0;

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;

    virtual expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const = 0;

    virtual expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const = 0;

    virtual Status
    Serialize(BinarySet& binset) const = 0;

    virtual Status
    Deserialize(const BinarySet& binset) = 0;

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const = 0;

    virtual int64_t
    Dim() const = 0;

    virtual int64_t
    Size() const = 0;

    virtual int64_t
    Count() const = 0;

    virtual std::string
    Type() const = 0;

    virtual ~IndexNode() {
    }
};

}  // namespace knowhere

#endif /* INDEX_NODE_H */
