###################################################################################################
# Test Faiss index
test_faiss_all: test_faiss_idmap test_faiss_ivf_flat test_faiss_ivf_sq8 test_faiss_ivf_pq test_faiss_hnsw

test_faiss_idmap:
	./benchmark_faiss --gtest_filter="Benchmark_faiss.TEST_IDMAP" | tee test_faiss_idmap.log
test_faiss_ivf_flat:
	./benchmark_faiss --gtest_filter="Benchmark_faiss.TEST_IVF_FLAT" | tee test_faiss_ivf_flat.log
test_faiss_ivf_sq8:
	./benchmark_faiss --gtest_filter="Benchmark_faiss.TEST_IVF_SQ8" | tee test_faiss_ivf_sq8.log
test_faiss_ivf_pq:
	./benchmark_faiss --gtest_filter="Benchmark_faiss.TEST_IVF_PQ" | tee test_faiss_ivf_pq.log
test_faiss_hnsw:
	./benchmark_faiss --gtest_filter="Benchmark_faiss.TEST_HNSW" | tee test_faiss_hnsw.log

###################################################################################################
# Test Knowhere index
test_knowhere_all: test_knowhere_idmap test_knowhere_ivf_flat_nm test_knowhere_ivf_sq8 test_knowhere_ivf_pq test_knowhere_ivf_hnsw test_knowhere_hnsw test_knowhere_annoy test_knowhere_rhnsw_flat test_knowhere_rhnsw_sq test_knowhere_rhnsw_pq

test_knowhere_idmap:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_IDMAP" | tee test_knowhere_idmap.log
test_knowhere_ivf_flat_nm:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_IVF_FLAT_NM" | tee test_knowhere_ivf_flat_nm.log
test_knowhere_ivf_sq8:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_IVF_SQ8" | tee test_knowhere_ivf_sq8.log
test_knowhere_ivf_pq:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_IVF_PQ" | tee test_knowhere_ivf_pq.log
test_knowhere_ivf_hnsw:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_IVF_HNSW" | tee test_knowhere_ivf_hnsw.log
test_knowhere_hnsw:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_HNSW" | tee test_knowhere_hnsw.log
test_knowhere_annoy:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_ANNOY" | tee test_knowhere_annoy.log
test_knowhere_rhnsw_flat:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_RHNSW_FLAT" | tee test_knowhere_rhnsw_flat.log
test_knowhere_rhnsw_sq:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_RHNSW_SQ" | tee test_knowhere_rhnsw_sq.log
test_knowhere_rhnsw_pq:
	./benchmark_knowhere_float --gtest_filter="Benchmark_knowhere_float.TEST_RHNSW_PQ" | tee test_knowhere_rhnsw_pq.log

###################################################################################################
# Test Knowhere index range
test_knowhere_range_all: test_knowhere_range_idmap test_knowhere_range_ivf_flat_nm test_knowhere_range_ivf_sq8 test_knowhere_range_ivf_pq test_knowhere_range_hnsw

test_knowhere_range_idmap:
	./benchmark_knowhere_float_range --gtest_filter="Benchmark_knowhere_float_range.TEST_IDMAP" | tee test_knowhere_range_idmap.log
test_knowhere_range_ivf_flat_nm:
	./benchmark_knowhere_float_range --gtest_filter="Benchmark_knowhere_float_range.TEST_IVF_FLAT_NM" | tee test_knowhere_range_ivf_flat_nm.log
test_knowhere_range_ivf_sq8:
	./benchmark_knowhere_float_range --gtest_filter="Benchmark_knowhere_float_range.TEST_IVF_SQ8" | tee test_knowhere_range_ivf_sq8.log
test_knowhere_range_ivf_pq:
	./benchmark_knowhere_float_range --gtest_filter="Benchmark_knowhere_float_range.TEST_IVF_PQ" | tee test_knowhere_range_ivf_pq.log
test_knowhere_range_hnsw:
	./benchmark_knowhere_float_range --gtest_filter="Benchmark_knowhere_float_range.TEST_HNSW" | tee test_knowhere_range_hnsw.log

###################################################################################################
# Test Knowhere binary index
test_knowhere_binary_all: test_knowhere_binary_idmap test_knowhere_binary_ivf_flat

test_knowhere_binary_idmap:
	./benchmark_knowhere_binary --gtest_filter="Benchmark_knowhere_binary.TEST_BINARY_IDMAP" | tee test_knowhere_binary_idmap.log
test_knowhere_binary_ivf_flat:
	./benchmark_knowhere_binary --gtest_filter="Benchmark_knowhere_binary.TEST_BINARY_IVF_FLAT" | tee test_knowhere_binary_ivf_flat.log

###################################################################################################
# Test Knowhere binary index range
test_knowhere_binary_range_all: test_knowhere_binary_range_idmap test_knowhere_binary_range_ivf_flat

test_knowhere_binary_range_idmap:
	./benchmark_knowhere_binary_range --gtest_filter="Benchmark_knowhere_binary_range.TEST_BINARY_IDMAP" | tee test_knowhere_binary_range_idmap.log
test_knowhere_binary_range_ivf_flat:
	./benchmark_knowhere_binary_range --gtest_filter="Benchmark_knowhere_binary_range.TEST_BINARY_IVF_FLAT" | tee test_knowhere_binary_range_ivf_flat.log

