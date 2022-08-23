import numpy as np


if __name__ == "__main__":
    x = np.random.randn(10, 128).astype(np.float32)
    x.tofile("./x.bin")
    y = np.random.randn(10000, 128).astype(np.float32)
    y.tofile("./y.bin")
