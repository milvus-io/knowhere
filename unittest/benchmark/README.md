## Benchmark description

 Binary Name | Description
-------------|------------
benchmark_faiss | benchmark to test FAISS `Query` for IDMAP and IVF serial index types
benchmark_knowhere_binary | benchmark to test knowhere `Query` for all binary index types
benchmark_knowhere_binary_range | benchmark to test knowhere `QueryByRange` for all supported binary index types
benchmark_knowhere_float | benchmark to test knowhere `Query` for all float index types
benchmark_knowhere_float_range | benchmark to test knowhere `QueryByRange` for all supported float index types
benchmark_knowhere_perf | benchmark to call knowhere `Query` periodically to simulate high CPU load

## How to run benchmark
 
#### Step 1:
Download latest HDF5 from:
  https://support.hdfgroup.org/ftp/HDF5/releases/

Use following commands to build HDF5 from source and install to "/usr/local/hdf5"
```bash
$ cd hdf5-1.13.1
$ ./configure --prefix=/usr/local/hdf5 --enable-fortran
$ make -j8
$ make check
$ sudo make install
```

#### Step 2:
Download HDF5 data files from:
  https://github.com/erikbern/ann-benchmarks

#### Step 3:
Update 'knowhere/unittest/CMakeLists.txt',
uncomment "#add_subdirectory(benchmark)".

#### Step 4:
Build Knowhere with unittest enabled: "./build.sh -t Release -u",
all benchmark binaries will be generated.

#### Step 5:
Put HDF5 data files into the directory 'output/unittest'.
Copy 'knowhere/unittest/benchmark/ref_log/Makefile' to 'output/unittest'.

#### Step 6:
Run benchmark test using following commands:
  - make test_faiss_all
  - make test_knowhere_all
  - make test_knowhere_range_all
  - make test_knowhere_binary_all
  - make test_knowhere_binary_range_all
