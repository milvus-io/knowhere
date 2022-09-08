import knowhere
import numpy as np
import json
import os

def test_async():
    id = os.getpid()
    print(id)
    idx = knowhere.CreateAsyncIndex("annoy")
    arr = np.random.uniform(1, 5, (10000, 128)).astype("float32")
    data = knowhere.ArrayToDataSet(arr)
    cmd = f"ps -p {id} -Tf | wc -l"
    print( os.popen(cmd).read() )
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
    cmd = f"ps -p {id} -Tf | wc -l"
    print( os.popen(cmd).read() )
    query_data = knowhere.ArrayToDataSet(
        arr[:1000, :]
        # np.random.uniform(1, 5, (1000, 128)).astype("float32")
    )
    idx.QueryAsync(query_data, cfg, knowhere.EmptyBitSetView())
    cmd = f"ps -p {id} -Tf | wc -l"
    print( os.popen(cmd).read() )
    ans = idx.Sync()
    idx = np.zeros((1000, 10), np.int32)
    dis = np.zeros((1000, 10), np.float32)
    knowhere.DumpResultDataSet(ans, dis, idx)
    print(idx)
    print(dis)


if __name__ == "__main__":
    test_async()
