cpu:
	cmake -B cmake_build . \
		-DCMAKE_BUILD_TYPE=Release \
		-DFAISS_ENABLE_PYTHON=OFF \
		-DFAISS_ENABLE_GPU=OFF \
		-DBUILD_TESTING=ON
	make -C cmake_build -j8

gpu:
	cmake -B cmake_build . \
		-DCMAKE_BUILD_TYPE=Release \
		-DFAISS_ENABLE_PYTHON=OFF \
		-DFAISS_ENABLE_GPU=ON \
		-DCUDAToolkit_ROOT=/usr/local/cuda \
		-DCMAKE_CUDA_ARCHITECTURES="75" \
		-DBUILD_TESTING=ON
	make -C cmake_build -j8

test:
	make -C cmake_build test

clean:
	rm -rf cmake_build

