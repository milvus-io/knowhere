#include <random>
#include "knowhere/dataset.h"

inline std::unique_ptr<knowhere::DataSet>
GenDataSet(int rows, int dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(0.0, 100.0);

    auto ds = std::make_unique<knowhere::DataSet>();
    ds->SetRows(rows);
    ds->SetDim(dim);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) ts[i] = distrib(rng);
    ds->SetTensor(ts);
    return ds;
}
