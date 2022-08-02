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


def CreateIndexDiskANN(index_name, index_prefix, metric_type, file_manager, simd_type="auto"):

    if simd_type not in ["auto", "avx512", "avx2", "avx", "sse4_2"]:
        raise ValueError("simd type only support auto avx512 avx2 avx sse4_2")

    SetSimdType(simd_type)

    if index_name == "diskann_f":
        return buildDiskANNf(index_prefix, metric_type, file_manager)
    if index_name == "diskann_i8":
        return buildDiskANNi8(index_prefix, metric_type, file_manager)
    if index_name == "diskann_ui8":
        return buildDiskANNui8(index_prefix, metric_type, file_manager)
    raise ValueError(
        """ index name only support 
            'diskann_f' 'diskann_i8' 'diskann_ui8'."""
    )


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
