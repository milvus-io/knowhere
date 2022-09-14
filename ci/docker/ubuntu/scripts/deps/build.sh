CMAKE_VERSION="v3.23"
apt-get update && apt-get install -y --no-install-recommends wget curl g++ gcc ca-certificates \
gpg make ccache python3-dev gfortran python3-setuptools swig pip \
libaio-dev libboost-program-options-dev \
&& wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
 2>/dev/null | gpg --dearmor - | tee \
/usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
&& echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main'  |  tee \
/etc/apt/sources.list.d/kitware.list >/dev/null \
&& apt update && apt install -y cmake \
&& apt-get remove --purge -y  \
&& rm -rf /var/lib/apt/lists/* \
&& wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.1.tar.gz \
&& tar zxvf v3.10.1.tar.gz && cd lapack-3.10.1/  \
&& cmake -B build -S . -DCMAKE_SKIP_RPATH=ON -DBUILD_SHARED_LIBS=ON \
-DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local \
-DCMAKE_Fortran_COMPILER=gfortran -DLAPACKE_WITH_TMG=ON -DCBLAS=OFF \
-DBUILD_DEPRECATED=OFF \
&& cmake --build build \
&& cmake --install build \
&& rm -r /usr/local/lib/libblas.* \
&& rm -r /usr/local/lib/liblapacke.* \
&& rm -r /usr/local/lib/pkgconfig/blas.* \
&& rm -r /usr/local/lib/pkgconfig/lapacke.* \
&& rm -r /usr/local/lib/cmake/lapacke* \
&& rm -r /usr/local/include/* \
&& cd .. && rm -rf lapack-3.10.1 && rm v3.10.1.tar.gz \
&& wget https://github.com/xianyi/OpenBLAS/archive/v0.3.21.tar.gz && \
tar zxvf v0.3.21.tar.gz && cd OpenBLAS-0.3.21 && \
make NO_STATIC=1 NO_LAPACK=1 NO_LAPACKE=1 NO_AFFINITY=1 USE_OPENMP=1 \
    TARGET=HASWELL DYNAMIC_ARCH=1 \
    NUM_THREADS=64 MAJOR_VERSION=3 libs shared && \
make PREFIX=/usr/local NUM_THREADS=64 MAJOR_VERSION=3 install && \
rm -f /usr/local/include/lapack* && \
ln -s /usr/local/lib/libopenblasp-r0.3.21.so /lib/libblas.so && \
ln -s /usr/local/lib/libopenblasp-r0.3.21.so /lib/libblas.so.3 && \
ln -s /usr/local/lib/pkgconfig/openblas.pc /usr/local/lib/pkgconfig/blas.pc && \
cd .. && rm -rf OpenBLAS-0.3.21 && rm v0.3.21.tar.gz \
&& pip install twine
