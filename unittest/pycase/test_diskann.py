import knowhere
import numpy as np
import json
import os
import shutil


def test_diskann():
    tmp_path = "test_diskann_tmp"
    try:
        shutil.rmtree(tmp_path)
    except:
        pass 
    os.mkdir(tmp_path)
    idx = knowhere.CreateIndexDiskANN("diskann_f", os.path.join(tmp_path, "index"), "L2", knowhere.EmptyFileManager())
    shape = np.array([10000, 128]).astype(np.int32)
    arr = np.random.uniform(1, 5, [10000, 128]).astype(np.float32)
    data_path = os.path.join(tmp_path, "diskann_data")
    try:
        os.remove(data_path) 
    except:
        pass
    with open(data_path, 'ab') as f:
        shape.tofile(f)
        arr.tofile(f)
    cfg = knowhere.CreateConfig(
        json.dumps(
            {
                "diskANN_build_config": {
                    "data_path": data_path,
                    "max_degree": 80,
                    "search_list_size": 128,
                    "search_dram_budget_gb": 1.0,
                    "build_dram_budget_gb": 2.0,
                    "num_threads": 16,
                    "pq_disk_bytes": 0
                },
                "diskANN_prepare_config": {
                    "num_threads": 16,
                    "num_nodes_to_cache": 100,
                    "warm_up": True,
                    "use_bfs_cache": False
                },
                "diskANN_query_config": {
                    "k": 1000,
                    "search_list_size": 10000,
                    "beamwidth": 1
                },
                "diskANN_query_by_range_config": {
                    "radius": 1.0,
                    "min_k": 1,
                    "max_k": 10000,
                    "beamwidth": 4
                }
            }
        )
    )
    idx.AddWithoutIds(None, cfg)
    idx.Prepare(cfg)
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
    try:
        shutil.rmtree(tmp_path)
    except:
        pass


if __name__ == "__main__":
    test_diskann()
