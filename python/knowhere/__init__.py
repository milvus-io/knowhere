from . import swigknowhere
import numpy as np


def CreateIndex(name):
    return swigknowhere.IndexWrap(name)


def ArrayToDataSet(arr):
    if arr.dtype == np.int32:
        return swigknowhere.Array2DataSetI(arr)
    if arr.dtype == np.float32:
        return swigknowhere.Array2DataSetF(arr)
    raise ValueError(
        """
        ArrayToDataSet only support numpy array dtype float32 and int32.
        """
    )


def DataSetToArray(ans):
    dim = swigknowhere.DataSet_Dim(ans)
    rows = swigknowhere.DataSet_Rows(ans)
    dis = np.zeros([rows, dim]).astype(np.float32)
    ids = np.zeros([rows, dim]).astype(np.int32)
    swigknowhere.DataSet2Array(ans, dis, ids)
    return dis, ids
