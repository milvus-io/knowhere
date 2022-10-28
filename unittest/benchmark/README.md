## Benchmark description

### Benchmark for 'Query' for binary index types

Following 1 benchmark read data from "sift-4096-hamming.hdf5"

1. benchmark_knowhere_binary

   benchmark to test knowhere `Query` for all binary index types

### Benchmark for 'QueryByRange' for binary index types

Following 1 benchmark read data from "sift-4096-hamming-range.hdf5"

1. benchmark_knowhere_binary_range

   benchmark to test knowhere `QueryByRange` with unified radius for all supported binary index types

### Benchmark for 'Query' for float index types

Following 3 benchmarks read data from "sift-128-eucliean.hdf5"

1. benchmark_faiss_float

   benchmark to test FAISS `Query` for IDMAP and IVF serial index types

2. benchmark_knowhere_float

   benchmark to test knowhere `Query` for all float index types

3. benchmark_knowhere_float_qps

   benchmark to test the QPS of knowhere `Query` for all float index types

### Benchmark for 'QueryByRange' with unified radius for float index types

Following 2 benchmark read data from "sift-128-eucliean-range.hdf5"

1. benchmark_knowhere_float_range

   benchmark to test knowhere `QueryByRange` with unified radius for all supported float index types

2. benchmark_knowhere_float_range_qps

   benchmark to test the QPS of knowhere `QueryByRange` with unified radius for all supported float index types

### Benchmark for 'QueryByRange' with different radius setting for float index types

Following 2 benchmark read data from "sift-128-eucliean-range-multi.hdf5"

1. benchmark_knowhere_float_range_multi

   benchmark to test knowhere `QueryByRange` with different radius setting for all supported float index types

2. benchmark_knowhere_float_range_multi_qps

   benchmark to test the QPS of knowhere `QueryByRange` with different radius setting for all supported float index types

## How to run benchmark
 
#### Step 1:
Run `prepare.sh`

#### Step 2:
Run benchmark according to `unittest/benchmark/hdf5/ref_log/Makefile`
Example commands:
  - make test_faiss_all
  - make test_knowhere_all
  - make test_knowhere_gpu
  - make test_knowhere_range_all
  - make test_knowhere_range_qps_all
  - make test_knowhere_range_multi_all
  - make test_knowhere_range_multi_qps_all
  - make test_knowhere_binary_all
  - make test_knowhere_binary_range_all

## Hardware Environment
The logs under directory 'ref_log' are tested under following hardware environment:
### CPU
  - Architecture:        x86_64
  - CPU op-mode(s):      32-bit, 64-bit
  - Byte Order:          Little Endian
  - CPU(s):              12
  - Thread(s) per core:  2
  - Core(s) per socket:  6
  - Model name:          Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
  - CPU MHz:             800.060
  - CPU max MHz:         4600.0000
  - CPU min MHz:         800.0000
  - L1d cache:           32K
  - L1i cache:           32K
  - L2 cache:            256K
  - L3 cache:            12288K
  - Flags:               sse sse2 ssse3 sse4_1 sse4_2 avx f16c avx2

### GPU
  - CUDA Driver Version / Runtime Version:         11.4 / 10.1
  - CUDA Capability Major/Minor version number:    7.5
  - Total amount of global memory:                 5945 MBytes (6233391104 bytes)
  - (22) Multiprocessors, ( 64) CUDA Cores/MP:     1408 CUDA Cores
  - GPU Max Clock rate:                            1830 MHz (1.83 GHz)
  - Memory Clock rate:                             4001 Mhz
  - Memory Bus Width:                              192-bit
  - L2 Cache Size:                                 1572864 bytes
  - Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  - Maximum Layered 1D Texture Size, (num) layers: 1D=(32768), 2048 layers
  - Maximum Layered 2D Texture Size, (num) layers: 2D=(32768, 32768), 2048 layers
  - Total amount of constant memory:               65536 bytes
  - Total amount of shared memory per block:       49152 bytes
  - Total number of registers available per block: 65536
  - Warp size:                                     32
  - Maximum number of threads per multiprocessor:  1024
  - Maximum number of threads per block:           1024
  - Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  - Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  - Maximum memory pitch:                          2147483647 bytes
  - Texture alignment:                             512 bytes
  - Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  - Run time limit on kernels:                     Yes
  - Integrated GPU sharing Host Memory:            No
  - Support host page-locked memory mapping:       Yes
  - Alignment requirement for Surfaces:            Yes
  - Device has ECC support:                        Disabled
  - Device supports Unified Addressing (UVA):      Yes
  - Device supports Compute Preemption:            Yes
  - Supports Cooperative Kernel Launch:            Yes
  - Supports MultiDevice Co-op Kernel Launch:      Yes
  - Device PCI Domain ID / Bus ID / location ID:   0 / 3 / 0

### Memory
  - 64G
