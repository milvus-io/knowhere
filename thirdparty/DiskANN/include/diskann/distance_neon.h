#pragma once
#include "distance.h"
#include <cstdint>
namespace diskann {
  class NeonDistanceL2Int8 : public Distance<int8_t> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const int8_t *x, const int8_t *y,
                                            uint32_t d) const;
  };

  class NeonDistanceL2Float : public Distance<float> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const float *x, const float *y,
                                            uint32_t d) const;
  };

  class NeonDistanceL2UInt8 : public Distance<uint8_t> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const uint8_t *x, const uint8_t *y,
                                            uint32_t d) const;
  };

  class NeonDistanceInnerProductFloat : public Distance<float> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const float *x, const float *y,
                                            uint32_t d) const;
  };

}  // namespace diskann
