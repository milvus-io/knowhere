###################################################################################################
# Test Knowhere index
test_ssnpp_all: test_ssnpp_ivf_flat_nm test_ssnpp_hnsw

test_ssnpp_ivf_flat_nm:
	./benchmark_ssnpp_float --gtest_filter="Benchmark_ssnpp_float.TEST_IVF_FLAT_NM" | tee test_ssnpp_ivf_flat_nm.log
test_ssnpp_hnsw:
	./benchmark_ssnpp_float --gtest_filter="Benchmark_ssnpp_float.TEST_HNSW" | tee test_ssnpp_hnsw.log

###################################################################################################
# Test Knowhere index range
test_ssnpp_range_all: test_ssnpp_range_ivf_flat_nm test_ssnpp_range_hnsw

test_ssnpp_range_ivf_flat_nm:
	./benchmark_ssnpp_float_range --gtest_filter="Benchmark_ssnpp_float_range.TEST_IVF_FLAT_NM" | tee test_ssnpp_range_ivf_flat_nm.log
test_ssnpp_range_hnsw:
	./benchmark_ssnpp_float_range --gtest_filter="Benchmark_ssnpp_float_range.TEST_HNSW" | tee test_ssnpp_range_hnsw.log
