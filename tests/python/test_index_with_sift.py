import knowhere
import pytest
import wget
import tarfile
import numpy as np
import json


def ivecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view("float32")


def download_sift():
    if not hasattr(download_sift, "done"):
        URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        response = wget.download(URL, "/tmp/sift.tar.gz")
        file = tarfile.open("/tmp/sift.tar.gz")
        file.extractall("/tmp")
        file.close()
        download_sift.done = True

    return download_sift.done


test_data = [
    (
        "FLAT",
        {
            "dim": 128,
            "k": 100,
            "metric_type": "L2",
        },
    ),
    (
        "ANNOY",
        {
            "dim": 128,
            "k": 100,
            "metric_type": "L2",
            "n_tree": 200,
            "search_k": 100,
        },
    ),
    (
        "IVFFLAT",
        {
            "dim": 128,
            "k": 100,
            "metric_type": "L2",
            "n_list": 1024,
            "nprobe": 128,
        },
    ),
    (
        "IVFSQ",
        {
            "dim": 128,
            "k": 100,
            "metric_type": "L2",
            "n_list": 1024,
            "nprobe": 128,
        },
    ),
    (
        "IVFPQ",
        {
            "dim": 128,
            "k": 100,
            "metric_type": "L2",
            "n_list": 1024,
            "nprobe": 128,
            "m": 4,
            "nbits": 8,
        },
    ),
    (
        "HNSW",
        {
            "dim": 128,
            "k": 100,
            "metric_type": "L2",
            "M": 1000,
            "efConstruction": 200,
            "ef": 32,
            "range_k": 100,
        },
    ),
]


@pytest.mark.parametrize("name,config", test_data)
def test_index_with_sift(recall, error, name, config):
    download_sift()
    xb = fvecs_read("/tmp/sift/sift_base.fvecs")
    xq = fvecs_read("/tmp/sift/sift_query.fvecs")
    ids_true = ivecs_read("/tmp/sift/sift_groundtruth.ivecs")

    idx = knowhere.CreateIndex(name)
    idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    ans, _ = idx.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(config),
        knowhere.GetNullBitSetView()
    )
    _, ids = knowhere.DataSetToArray(ans)
    assert recall(ids_true, ids) >= 0.99
