from .swigknowhere import *
import numpy as np


def CreateIndex(index_name, simd_type="auto"):

    if simd_type not in ["auto", "avx512", "avx2", "avx", "sse4_2"]:
        raise ValueError("simd type only support auto avx512 avx2 avx sse4_2")

    SetSimdType(simd_type)

    if index_name == "bin_flat":
        return BinaryIDMAP()
    if index_name == "bin_ivf_flat":
        return BinaryIVF()
    if index_name == "flat":
        return IDMAP()
    if index_name == "ivf_flat":
        return IVF()
    if index_name == "ivf_pq":
        return IVFPQ()
    if index_name == "ivf_sq8":
        return IVFSQ()
    if index_name == "hnsw":
        return IndexHNSW()
    if index_name == "annoy":
        return IndexAnnoy()
    if index_name == "gpu_flat":
        return GPUIDMAP(0)
    if index_name == "gpu_ivf_flat":
        return GPUIVF(0)
    if index_name == "gpu_ivf_pq":
        return GPUIVFPQ(0)
    if index_name == "gpu_ivf_sq8":
        return GPUIVFSQ(0)
    raise ValueError(
        """ index name only support 
            'bin_flat' 'bin_ivf_flat' 'flat' 'ivf_flat' 'ivf_pq' 'ivf_sq8' 'hnsw' 'annoy'
            'gpu_flat' 'gpu_ivf_flat', 'gpu_ivf_pq', 'gpu_ivf_sq8'."""
    )


def CreateIndexDiskANN(index_name, index_prefix, metric_type, simd_type="auto"):

    if simd_type not in ["auto", "avx512", "avx2", "avx", "sse4_2"]:
        raise ValueError("simd type only support auto avx512 avx2 avx sse4_2")

    SetSimdType(simd_type)

    if index_name == "diskann_f":
        return buildDiskANNf(index_prefix, metric_type)
    raise ValueError(
        """ index name only support 'diskann_f'. """
    )


def CreateAsyncIndex(index_name, index_prefix="", metric_type="", simd_type="auto"):
    if simd_type not in ["auto", "avx512", "avx2", "avx", "sse4_2"]:
        raise ValueError("simd type only support auto avx512 avx2 avx sse4_2")

    SetSimdType(simd_type)

    if index_name not in ["bin_flat", "bin_ivf_flat", "flat", "ivf_flat", "ivf_pq", "ivf_sq8",
                          "hnsw", "annoy", "gpu_flat", "gpu_ivf_flat",
                          "gpu_ivf_pq", "gpu_ivf_sq8", "diskann_f"]:
        raise ValueError(
            """ index name only support
            'bin_flat', 'bin_ivf_flat', 'flat', 'ivf_flat', 'ivf_pq', 'ivf_sq8',
            'hnsw', 'annoy', 'gpu_flat', 'gpu_ivf_flat',
            'gpu_ivf_pq', 'gpu_ivf_sq8', 'diskann_f'."""
        )

    if index_name == "diskann_f":
        if index_prefix == "":
            raise ValueError("Must pass index_prefix to DiskANN")
        if metric_type == "":
            raise ValueError("Must pass metric_type to DiskANN")
        return AsyncIndex(index_name, index_prefix, metric_type)
    else:
        return AsyncIndex(index_name)


class GpuContext:
    def __init__(
        self, dev_id=0, pin_mem=200 * 1024 * 1024, temp_mem=300 * 1024 * 1024, res_num=2
    ):
        InitGpuResource(dev_id, pin_mem, temp_mem, res_num)

    def __del__(self):
        ReleaseGpuResource()


def ArrayToDataSet(arr):
    if arr.dtype == np.int32:
        return ArrayToDataSetInt(arr)
    if arr.dtype == np.float32:
        return ArrayToDataSetFloat(arr)
    raise ValueError(
        """
        ArrayToDataSet only support numpy array dtype float32 and int32.
        """
    )


def UnpackRangeResults(results, nq):
    lims = np.zeros(
        [
            nq + 1,
        ],
        dtype=np.int32,
    )
    DumpRangeResultLimits(results, lims)
    dis = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.float32,
    )
    DumpRangeResultDis(results, dis)
    ids = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.int32,
    )
    DumpRangeResultIds(results, ids)
 
    dis_list = []
    ids_list = []

    for idx in range(nq):
        dis_list.append(dis[lims[idx] : lims[idx + 1]])
        ids_list.append(ids[lims[idx] : lims[idx + 1]])

    return ids_list, dis_list


def ReadFromFBIN(filename):
    with open(filename, 'rb') as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        arr = np.fromfile(f, dtype=np.float32)
        arr.resize(n, dim)
    return arr