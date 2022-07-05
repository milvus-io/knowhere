FROM --platform=arm64 ubuntu:20.04

RUN apt-get update  \
    && apt-get install -y g++ gcc make ccache python3-dev gfortran \
      python3-setuptools swig libopenblas-dev \
    && apt-get remove --purge -y  \
    && rm -rf /var/lib/apt/lists/*