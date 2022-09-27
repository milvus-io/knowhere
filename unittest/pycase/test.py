import json
import os
import shutil
import knowhere
import numpy as np


def test_boundary_beamwidth():
    def build_config(data_path, pq_code_budget_gb, max_degree=24, num_threads=8):
        # returns the config for build
        cfg = {
            "data_path": data_path,
            "max_degree": max_degree,
            "search_list_size": 64,
            "pq_code_budget_gb": pq_code_budget_gb,

            "build_dram_budget_gb": 32.0,
            "num_threads": num_threads,
            "disk_pq_dims": 0,
            "accelerate_build": False
        }

        return cfg

    def prepare_config(use_bfs_cache=False, search_cache_budget_gb=0.006839633):
        cfg = {
            "num_threads": 8,
            # "num_nodes_to_cache": constants.NUM_NODES_TO_CACHE,
            "search_cache_budget_gb": search_cache_budget_gb,
            "warm_up": False,
            "use_bfs_cache": use_bfs_cache
        }
        return cfg

    def query_config(search_list_size_i, beamwidth=16):
        cfg = {
            "k": 100,
            "search_list_size": search_list_size_i,
            "beamwidth": beamwidth
        }
        return cfg

    beamwidth = 128
    tmp_path = "test_diskann_query"
    index_prefix = create_index_dir(tmp_path)
    dim = 128
    arr, data_path = knn_write_data(tmp_path, dim)
    nq = 1
    top_k = 100
    # search_cache_budget_gb = 0.002279877667
    search_cache_budget_gb = 0.000000001
    # index
    metric_type = "l2"
    idx = knowhere.CreateAsyncIndex("diskann_f", index_prefix, metric_type)
    search_list_size = 150

    search_dram_budget_gb = 0.03
    build_config = build_config(data_path, search_dram_budget_gb)
    prepare_config = prepare_config(search_cache_budget_gb=search_cache_budget_gb)
    query_config = query_config(search_list_size, beamwidth)
    cfg = {"diskANN_build_config": build_config, "diskANN_prepare_config": prepare_config,
           "diskANN_query_config": query_config}
    cfg = knowhere.CreateConfig(json.dumps(cfg))
    # start to build index
    idx.AddWithoutIds(None, cfg)
    idx.Prepare(cfg)
    query_data = knowhere.ArrayToDataSet(
        arr[:nq, :]
    )

    idx.QueryAsync(query_data, cfg, knowhere.EmptyBitSetView())
    # ans2 = idx.QueryAsync(query_data, cfg, knowhere.EmptyBitSetView())
    # ans3 = idx.QueryAsync(query_data, cfg, knowhere.EmptyBitSetView())
    ans1 = idx.Sync()
    result_ids = np.zeros((nq, top_k), np.int32)
    result_dis = np.zeros((nq, top_k), np.float32)
    knowhere.DumpResultDataSet(ans1, result_dis, result_ids)

    print(ans1)
    # print(ans2)
    # print(ans3)
    print(result_ids)
    return check_answer(result_ids, nq, top_k)


def create_index_dir(tmp_path="test_diskann_tmp"):
    """
    create a clean tmp_path folder,if there is a folder, it will be deleted and created again
    """
    if os.path.exists(tmp_path):
        try:
            shutil.rmtree(tmp_path)
        except:
            pass
    os.makedirs(tmp_path)
    index_prefix = os.path.join(tmp_path, "index")
    return index_prefix


def knn_write_data(bin_files_dir, dims, data_volume=10):
    """
    write random data in the dims dimension in tmp_path
    """
    shape = np.array([data_volume, dims]).astype(np.uint32)
    arr = np.random.uniform(1, 5, [data_volume, dims]).astype(np.float32)
    data_path = os.path.join(bin_files_dir, "diskann_data")
    with open(data_path, 'w') as f:
        shape.tofile(f)
        arr.tofile(f)
    f.close()
    return arr, data_path


def check_answer(result_ids, nq, top_k):
    zero = np.zeros((nq, top_k), np.int32)
    minus_one = np.ones((nq, top_k), np.int32) * -1
    if (result_ids == zero).all():
        return False
    if (result_ids == minus_one).all():
        return False
    return True


if __name__ == '__main__':
    print(test_boundary_beamwidth())
