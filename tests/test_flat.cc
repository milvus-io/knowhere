#include <knowhere/knowhere.h>

#include <fstream>

void
flat() {
    auto idx = knowhere::IndexFactory::Instance().Create("GPUIVFSQ");

    knowhere::DataSet xb;
    xb.SetRows(10000);
    xb.SetDim(128);

    {
        std::fstream f("./y.bin", f.in | f.binary);
        f.seekg(0, f.end);
        size_t s_size = f.tellg();
        f.seekg(0, f.beg);
        float* data = new float[s_size / sizeof(float)];
        f.read((char*)data, s_size);
        xb.SetTensor(data);
    }
    auto cfg = idx->CreateConfig();
    knowhere::Json json = knowhere::Json::parse(
        R"({"dim": 128, "metric_type": "L2", "k": 10, "radius": 100.0 , "nlist": 100, "nprobe": 80, "gpu_id": 0, "m": 4, "nbits": 8 })");
    auto r = knowhere::Config::Load(*cfg, json, knowhere::PARAM_TYPE::TRAIN);
    assert(r == knowhere::Error::success);
    idx->Train(xb, *cfg);
    idx->Add(xb, *cfg);
    knowhere::DataSet xq;
    xq.SetRows(10);
    xq.SetDim(128);
    {
        std::fstream f("./x.bin", f.in | f.binary);
        f.seekg(0, f.end);
        size_t s_size = f.tellg();
        f.seekg(0, f.beg);
        float* data = new float[s_size / sizeof(float)];
        f.read((char*)data, s_size);
        xq.SetTensor(data);
    }
    r = knowhere::Config::Load(*cfg, json, knowhere::PARAM_TYPE::SEARCH);
    assert(r == knowhere::Error::success);
    for (int i = 0; i < 100000; ++i) {
        auto res = idx->Search(xq, *cfg, nullptr);
        if (!res.has_value())
            std::cout << (int)res.error() << std::endl;
        {
            std::fstream f("./z.bin", f.out | f.binary);
            f.write((const char*)res.value()->GetIds(), 10 * 10 * sizeof(int64_t));
        }
    }
}

int
main() {
    flat();

    return 0;
}
