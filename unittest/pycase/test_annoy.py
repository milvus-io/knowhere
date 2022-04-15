import knowhere
import numpy as np
import json


def test_annoy():
    idx = knowhere.CreateIndex("annoy")
    arr = np.random.uniform(1, 5, (10000, 128)).astype("float32")
    data = knowhere.ArrayToDataSet(arr)

    cfg = knowhere.CreateConfig(
        json.dumps(
            {
                "dim": 128,
                "k": 10,
                "n_trees": 4,
                "search_k": 100,
                "metric_type": "L2",
            }
        )
    )
    idx.BuildAll(data, cfg)
    query_data = knowhere.ArrayToDataSet(
        arr[:1000, :]
        # np.random.uniform(1, 5, (1000, 128)).astype("float32")
    )
    ans = idx.Query(query_data, cfg, knowhere.EmptyBitSetView())
    idx = np.zeros((1000, 10), np.int32)
    dis = np.zeros((1000, 10), np.float32)
    knowhere.DumpResultDataSet(ans, dis, idx)
    print(idx)
    print(dis)


if __name__ == "__main__":
    test_annoy()
