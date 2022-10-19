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
    (
        "IVFFLAT",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "n_list": 1024,
            "nprobe": 1024,
        },
    ),
    """
    (
        "IVFSQ",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "n_list": 1024,
            "nprobe": 1024,
        },
    ),
    (
        "IVFPQ",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "n_list": 1024,
            "nprobe": 1024,
            "m": 32,
            "nbits": 32,
        },
    ),
    """(
        "HNSW",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "M": 10000,
            "efConstruction": 200,
            "ef": 32,
            "range_k": 100,
        },
    ),
]


@pytest.mark.parametrize("name,config", test_data)
def test_index(gen_data, faiss_ans, recall, error, name, config):
    print(name, config)
    idx = knowhere.CreateIndex(name)
    xb, xq = gen_data(10000, 100, 256)

    idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    ans = idx.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(config),
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, config["metric_type"], config["k"])
    assert recall(f_ids, k_ids) >= 0.99
    assert error(f_dis, f_dis) <= 0.01
