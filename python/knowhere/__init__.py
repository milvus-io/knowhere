from .swigknowhere import *


def CreateIndex(index_name):
    if index_name == "annoy":
        return IndexAnnoy()
    if index_name == "ivf":
        return IVF()
    if index_name == "ivfsq":
        return IVFSQ()
    if index_name == "hnsw":
        return IndexHNSW()
    if index_name == "gpu_ivf":
        return GPUIVF(-1)
    if index_name == "gpu_ivfpq":
        return GPUIVFPQ(-1)
    if index_name == "gpu_ivfsq":
        return GPUIVFSQ(-1)
    raise ValueError(
        """ index name only support 
            'annoy' 'ivf' 'ivfsq' 'hnsw'
            'gpu_ivf', 'gpu_ivfsq', 'gpu_ivfpq'."""
    )


class GpuContext:
    def __init__(
        self, dev_id=0, pin_mem=200 * 1024 * 1024, temp_mem=300 * 1024 * 1024, res_num=2
    ):
        InitGpuResource(dev_id, pin_mem, temp_mem, res_num)

    def __del__(self):
        ReleaseGpuResource()
