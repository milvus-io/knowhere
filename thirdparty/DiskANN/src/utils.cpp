// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

#include <stdio.h>

const uint32_t MAX_REQUEST_SIZE = 1024 * 1024 * 1024;  // 64MB
const uint32_t MAX_SIMULTANEOUS_READ_REQUESTS = 128;

#ifdef _WINDOWS
#include <intrin.h>

// Taken from:
// https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
bool cpuHasAvxSupport() {
  bool avxSupported = false;

  // Checking for AVX requires 3 things:
  // 1) CPUID indicates that the OS uses XSAVE and XRSTORE
  //     instructions (allowing saving YMM registers on context
  //     switch)
  // 2) CPUID indicates support for AVX
  // 3) XGETBV indicates the AVX registers will be saved and
  //     restored on context switch
  //
  // Note that XGETBV is only available on 686 or later CPUs, so
  // the instruction needs to be conditionally run.
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);

  bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
  bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;

  if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    avxSupported = (xcrFeatureMask & 0x6) || false;
  }

  return avxSupported;
}

bool cpuHasAvx2Support() {
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int n = cpuInfo[0];
  if (n >= 7) {
    __cpuidex(cpuInfo, 7, 0);
    static int avx2Mask = 0x20;
    return (cpuInfo[1] & avx2Mask) > 0;
  }
  return false;
}
#endif

#ifdef _WINDOWS
bool AvxSupportedCPU = cpuHasAvxSupport();
bool Avx2SupportedCPU = cpuHasAvx2Support();
#else
bool Avx2SupportedCPU = true;
bool AvxSupportedCPU = false;
#endif

namespace diskann {
  // Get the right distance function for the given metric.
  template<>
  diskann::Distance<float>* get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      if (Avx2SupportedCPU) {
        LOG(INFO) << "L2: Using AVX2 distance computation DistanceL2Float";
        return new diskann::DistanceL2Float();
      } else if (AvxSupportedCPU) {
        LOG(WARNING) << "L2: AVX2 not supported. Using AVX distance computation";
        return new diskann::AVXDistanceL2Float();
      } else {
        LOG(WARNING) << "L2: Older CPU. Using slow distance computation";
        return new diskann::SlowDistanceL2Float();
      }
    } else if (m == diskann::Metric::COSINE) {   
      LOG(INFO) << "Cosine: Using either AVX or AVX2 implementation";
      return new diskann::DistanceCosineFloat();
    } else if (m == diskann::Metric::INNER_PRODUCT) {
      LOG(INFO) << "Inner product: Using AVX2 implementation AVXDistanceInnerProductFloat";
      return new diskann::AVXDistanceInnerProductFloat();
    } else if (m == diskann::Metric::FAST_L2) {
      LOG(INFO) << "Fast_L2: Using AVX2 implementation with norm memoization DistanceFastL2<float>";
      return new diskann::DistanceFastL2<float>();
    } else {
      std::stringstream stream;
      stream << "Only L2, cosine, and inner product supported for floating "
                "point vectors as of now. Email "
                "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                "for any other metric."
             << std::endl;
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<int8_t>* get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      if (Avx2SupportedCPU) {
        LOG(INFO) << "Using AVX2 distance computation DistanceL2Int8.";
        return new diskann::DistanceL2Int8();
      } else if (AvxSupportedCPU) {
        LOG(WARNING) << "AVX2 not supported. Using AVX distance computation";
        return new diskann::AVXDistanceL2Int8();
      } else {
        LOG(WARNING) << "Older CPU. Using slow distance computation SlowDistanceL2Int<int8_t>.";
        return new diskann::SlowDistanceL2Int<int8_t>();
      }
    } else if (m == diskann::Metric::COSINE) {
      LOG(INFO) << "Using either AVX or AVX2 for Cosine similarity DistanceCosineInt8.";
      return new diskann::DistanceCosineInt8();
    } else {
      std::stringstream stream;
      stream << "Only L2 and cosine supported for signed byte vectors as of "
                "now. Email "
                "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                "for any other metric."
             << std::endl;
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<uint8_t>* get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
#ifdef _WINDOWS
      LOG(WARNING)
          << "WARNING: AVX/AVX2 distance function not defined for Uint8. Using "
             "slow version. "
             "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
          << std::endl;
#endif
      return new diskann::DistanceL2UInt8();
    } else if (m == diskann::Metric::COSINE) {
      LOG(WARNING) << "AVX/AVX2 distance function not defined for Uint8. Using "
                      "slow version SlowDistanceCosineUint8() "
                      "Contact gopalsr@microsoft.com if you need AVX/AVX2 support.";
      return new diskann::SlowDistanceCosineUInt8();
    } else {
      std::stringstream stream;
      stream << "Only L2 and cosine supported for unsigned byte vectors as of "
                "now. Email "
                "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                "for any other metric."
             << std::endl;
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  void block_convert(std::ofstream& writr, std::ifstream& readr,
                     float* read_buf, _u64 npts, _u64 ndims) {
    readr.read((char*) read_buf, npts * ndims * sizeof(float));
    _u32 ndims_u32 = (_u32) ndims;
#pragma omp parallel for
    for (_s64 i = 0; i < (_s64) npts; i++) {
      float norm_pt = std::numeric_limits<float>::epsilon();
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        norm_pt +=
            *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
      }
      norm_pt = std::sqrt(norm_pt);
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
      }
    }
    writr.write((char*) read_buf, npts * ndims * sizeof(float));
  }

  void normalize_data_file(const std::string& inFileName,
                           const std::string& outFileName) {
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    int npts_s32, ndims_s32;
    readr.read((char*) &npts_s32, sizeof(_s32));
    readr.read((char*) &ndims_s32, sizeof(_s32));

    writr.write((char*) &npts_s32, sizeof(_s32));
    writr.write((char*) &ndims_s32, sizeof(_s32));

    _u64 npts = (_u64) npts_s32, ndims = (_u64) ndims_s32;
    LOG(DEBUG) << "Normalizing FLOAT vectors in file: " << inFileName
               << "Dataset: #pts = " << npts << ", # dims = " << ndims;

    _u64 blk_size = 131072;
    _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
    LOG(DEBUG) << "# blks: " << nblks;

    float* read_buf = new float[npts * ndims];
    for (_u64 i = 0; i < nblks; i++) {
      _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
      block_convert(writr, readr, read_buf, cblk_size, ndims);
    }
    delete[] read_buf;

    LOG(DEBUG) << "Wrote normalized points to file: " << outFileName;
  }

}  // namespace diskann