### To run this FAISS benchmark, please follow these steps:
 
#### Step 1:
Download the HDF5 source from:
  https://support.hdfgroup.org/ftp/HDF5/releases/
and build/install to "/usr/local/hdf5".

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
