import knowhere
import json
import pytest

test_data = [
    (
        "FLAT",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
        },
    ),
    (
        "ANNOY",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "n_tree": 10000,
            "search_k": 100000,
        },
    ),
]
@pytest.mark.parametrize("name,config", test_data)
def test_save_and_load(gen_data, faiss_ans, recall, error, name, config):
    # simple load and save not work for ivf nm
    print(name, config)
    build_idx = knowhere.CreateIndex(name)
    xb, xq = gen_data(10000, 100, 256)

    build_idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    binset = knowhere.GetIndexBinarySet()
    build_idx.Save(binset)
    search_idx = knowhere.CreateIndex(name)
    search_idx.Load(binset)
    ans = search_idx.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(config),
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, config["metric_type"], config["k"])
    assert recall(f_ids, k_ids) >= 0.99
    assert error(f_dis, f_dis) <= 0.01
