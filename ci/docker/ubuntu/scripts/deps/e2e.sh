apt-get update && apt-get install -y --no-install-recommends wget curl g++ gcc make ca-certificates \
 python3-dev gfortran python3-setuptools swig liblapack-dev pip git vim \
 libaio-dev libboost-all-dev \
&& apt-get remove --purge -y  \
&& rm -rf /var/lib/apt/lists/* \
&& wget https://github.com/xianyi/OpenBLAS/archive/v0.3.21.tar.gz && \
tar zxvf v0.3.21.tar.gz && cd OpenBLAS-0.3.21 && \
make NO_STATIC=1 NO_LAPACK=1 NO_LAPACKE=1 NO_CBLAS=1 NO_AFFINITY=1 USE_OPENMP=1 \
    TARGET=HASWELL DYNAMIC_ARCH=1 \
    NUM_THREADS=64 MAJOR_VERSION=3 libs shared && \
make PREFIX=/usr/local NUM_THREADS=64 MAJOR_VERSION=3 install && \
rm -f /usr/local/include/cblas.h /usr/local/include/lapack* && \
cd .. && rm -rf OpenBLAS-0.3.21 && rm v0.3.21.tar.gz 
