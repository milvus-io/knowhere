apt-get update && apt-get install -y --no-install-recommends wget curl g++ gcc ca-certificates \
 python3-dev gfortran python3-setuptools swig libopenblas-dev pip git vim \
&& apt-get remove --purge -y  \
&& rm -rf /var/lib/apt/lists/* 