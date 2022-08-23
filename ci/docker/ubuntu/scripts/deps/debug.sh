CMAKE_VERSION="v3.23"
CMAKE_TAR="cmake-3.23.0-linux-x86_64.tar.gz"
HDF_VERSION="1.13.2"
apt-get update && apt-get install -y --no-install-recommends wget curl g++ gcc ca-certificates \
make ccache python3-dev gfortran python3-setuptools swig libopenblas-dev pip git vim \
libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev \
&& apt-get remove --purge -y  \
&& rm -rf /var/lib/apt/lists/* \
&& cd /tmp && wget  --tries=3 --retry-connrefused "https://cmake.org/files/${CMAKE_VERSION}/${CMAKE_TAR}" \
&& tar --strip-components=1 -xz -C /usr/local -f ${CMAKE_TAR} \
&& rm -f ${CMAKE_TAR} \

wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-${HDF_VERSION}/src/hdf5-${HDF_VERSION}.tar.gz && \
    tar xvfz hdf5-${HDF_VERSION}.tar.gz && cd hdf5-${HDF_VERSION} && \
    ./configure --prefix=/usr/local/hdf5 --enable-fortran && \
    make -j && make install && \
    cd .. && rm -rf hdf5-${HDF_VERSION} && rm hdf5-${HDF_VERSION}.tar.gz