CMAKE_VERSION="v3.23"
CMAKE_TAR="cmake-3.23.0-linux-x86_64.tar.gz"
apt-get update && apt-get install -y --no-install-recommends wget curl g++ gcc ca-certificates \
make ccache python3-dev gfortran python3-setuptools swig libopenblas-dev pip git vim \
&& apt-get remove --purge -y  \
&& rm -rf /var/lib/apt/lists/* \
&& cd /tmp && wget  --tries=3 --retry-connrefused "https://cmake.org/files/${CMAKE_VERSION}/${CMAKE_TAR}" \
&& tar --strip-components=1 -xz -C /usr/local -f ${CMAKE_TAR} \
&& rm -f ${CMAKE_TAR} \

wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.1/src/hdf5-1.13.1.tar.gz && \
    tar xvfz hdf5-1.13.1.tar.gz && cd hdf5-1.13.1 && \
    ./configure --prefix=/usr/local/hdf5 --enable-fortran && \
    make -j && make install && \
    cd .. && rm -rf hdf5-1.13.1 && rm hdf5-1.13.1.tar.gz