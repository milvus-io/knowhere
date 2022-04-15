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
    raise ValueError("index name only support 'annoy' 'ivf' 'ivfsq' 'hnsw'.")
