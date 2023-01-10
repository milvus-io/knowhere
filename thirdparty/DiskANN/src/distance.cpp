// TODO
// CHECK COSINE ON LINUX

#if defined(_WINDOWS)
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <intrin.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

#include "diskann/simd_utils.h"
#include "diskann/cosine_similarity.h"
#include <iostream>
#include <math.h>

#include "diskann/distance.h"
#include "diskann/logger.h"
#include "diskann/ann_exception.h"

namespace diskann {
#if defined(__x86_64__)
  namespace {
#define ALIGNED(x) __attribute__((aligned(x)))

    // reads 0 <= d < 4 floats as __m128
    inline __m128 masked_read(int d, const float *x) {
      ALIGNED(16) float buf[4] = {0, 0, 0, 0};
      switch (d) {
        case 3:
          buf[2] = x[2];
        case 2:
          buf[1] = x[1];
        case 1:
          buf[0] = x[0];
      }
      return _mm_load_ps(buf);
      // cannot use AVX2 _mm_mask_set1_epi32
    }
  }  // namespace
#endif
  // Cosine similarity.
  float DistanceCosineInt8::compare(const int8_t *a, const int8_t *b,
                                    uint32_t length) const {
#ifdef _WINDOWS
    return diskann::CosineSimilarity2<int8_t>(a, b, length);
#else
    int magA = 0, magB = 0, scalarProduct = 0;
    for (uint32_t i = 0; i < length; i++) {
      magA += ((int32_t) a[i]) * ((int32_t) a[i]);
      magB += ((int32_t) b[i]) * ((int32_t) b[i]);
      scalarProduct += ((int32_t) a[i]) * ((int32_t) b[i]);
    }
    // similarity == 1-cosine distance
    return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
#endif
  }

  float DistanceCosineFloat::compare(const float *a, const float *b,
                                     uint32_t length) const {
#ifdef _WINDOWS
    return diskann::CosineSimilarity2<float>(a, b, length);
#else
    float magA = 0, magB = 0, scalarProduct = 0;
    for (uint32_t i = 0; i < length; i++) {
      magA += (a[i]) * (a[i]);
      magB += (b[i]) * (b[i]);
      scalarProduct += (a[i]) * (b[i]);
    }
    // similarity == 1-cosine distance
    return 1.0f - (scalarProduct / (sqrt(magA) * sqrt(magB)));
#endif
  }

  float SlowDistanceCosineUInt8::compare(const uint8_t *a, const uint8_t *b,
                                         uint32_t length) const {
    int magA = 0, magB = 0, scalarProduct = 0;
    for (uint32_t i = 0; i < length; i++) {
      magA += ((uint32_t) a[i]) * ((uint32_t) a[i]);
      magB += ((uint32_t) b[i]) * ((uint32_t) b[i]);
      scalarProduct += ((uint32_t) a[i]) * ((uint32_t) b[i]);
    }
    // similarity == 1-cosine distance
    return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
  }

  // L2 distance functions.
  float DistanceL2Int8::compare(const int8_t *a, const int8_t *b,
                                uint32_t size) const {
    int32_t result = 0;

#ifdef _WINDOWS
#ifdef USE_AVX2
    __m256 r = _mm256_setzero_ps();
    char  *pX = (char *) a, *pY = (char *) b;
    while (size >= 32) {
      __m256i r1 = _mm256_subs_epi8(_mm256_loadu_si256((__m256i *) pX),
                                    _mm256_loadu_si256((__m256i *) pY));
      r = _mm256_add_ps(r, _mm256_mul_epi8(r1, r1));
      pX += 32;
      pY += 32;
      size -= 32;
    }
    while (size > 0) {
      __m128i r2 = _mm_subs_epi8(_mm_loadu_si128((__m128i *) pX),
                                 _mm_loadu_si128((__m128i *) pY));
      r = _mm256_add_ps(r, _mm256_mul32_pi8(r2, r2));
      pX += 4;
      pY += 4;
      size -= 4;
    }
    r = _mm256_hadd_ps(_mm256_hadd_ps(r, r), r);
    return r.m256_f32[0] + r.m256_f32[4];
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
    for (_s32 i = 0; i < (_s32) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
#endif
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
    for (int32_t i = 0; i < (int32_t) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
#endif
  }

  float DistanceL2UInt8::compare(const uint8_t *a, const uint8_t *b,
                                 uint32_t size) const {
    uint32_t result = 0;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
#endif
    for (int32_t i = 0; i < (int32_t) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
  }

  float DistanceL2Float::compare(const float *x, const float *y,
                                 uint32_t d) const {
#if defined(__x86_64__)
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
      __m256 mx = _mm256_loadu_ps(x);
      x += 8;
      __m256 my = _mm256_loadu_ps(y);
      y += 8;
      const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
      msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1, a_m_b1));
      d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
      __m128 mx = _mm_loadu_ps(x);
      x += 4;
      __m128 my = _mm_loadu_ps(y);
      y += 4;
      const __m128 a_m_b1 = _mm_sub_ps(mx, my);
      msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
      d -= 4;
    }

    if (d > 0) {
      __m128 mx = masked_read(d, x);
      __m128 my = masked_read(d, y);
      __m128 a_m_b1 = _mm_sub_ps(mx, my);
      msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
#else
    float result = 0.0f;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(x, y : 32)
#endif
    for (int32_t i = 0; i < (int32_t) d; i++) {
      result += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return result;
#endif
  }

  float SlowDistanceL2Float::compare(const float *a, const float *b,
                                     uint32_t length) const {
    float result = 0.0f;
    for (uint32_t i = 0; i < length; i++) {
      result += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return result;
  }

#ifdef _WINDOWS
  float AVXDistanceL2Int8::compare(const int8_t *a, const int8_t *b,
                                   uint32_t length) const {
    __m128  r = _mm_setzero_ps();
    __m128i r1;
    while (length >= 16) {
      r1 = _mm_subs_epi8(_mm_load_si128((__m128i *) a),
                         _mm_load_si128((__m128i *) b));
      r = _mm_add_ps(r, _mm_mul_epi8(r1));
      a += 16;
      b += 16;
      length -= 16;
    }
    r = _mm_hadd_ps(_mm_hadd_ps(r, r), r);
    float res = r.m128_f32[0];

    if (length >= 8) {
      __m128  r2 = _mm_setzero_ps();
      __m128i r3 = _mm_subs_epi8(_mm_load_si128((__m128i *) (a - 8)),
                                 _mm_load_si128((__m128i *) (b - 8)));
      r2 = _mm_add_ps(r2, _mm_mulhi_epi8(r3));
      a += 8;
      b += 8;
      length -= 8;
      r2 = _mm_hadd_ps(_mm_hadd_ps(r2, r2), r2);
      res += r2.m128_f32[0];
    }

    if (length >= 4) {
      __m128  r2 = _mm_setzero_ps();
      __m128i r3 = _mm_subs_epi8(_mm_load_si128((__m128i *) (a - 12)),
                                 _mm_load_si128((__m128i *) (b - 12)));
      r2 = _mm_add_ps(r2, _mm_mulhi_epi8_shift32(r3));
      res += r2.m128_f32[0] + r2.m128_f32[1];
    }

    return res;
  }

  float AVXDistanceL2Float::compare(const float *a, const float *b,
                                    uint32_t length) const {
    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (length >= 4) {
      v1 = _mm_loadu_ps(a);
      a += 4;
      v2 = _mm_loadu_ps(b);
      b += 4;
      diff = _mm_sub_ps(v1, v2);
      sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
      length -= 4;
    }

    return sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] +
           sum.m128_f32[3];
  }
#else
  float AVXDistanceL2Int8::compare(const int8_t *, const int8_t *,
                                   uint32_t) const {
    return 0;
  }
  float AVXDistanceL2Float::compare(const float *, const float *,
                                    uint32_t) const {
    return 0;
  }
#endif

  template<typename T>
  float DistanceInnerProduct<T>::inner_product(const T *a, const T *b,
                                               unsigned size) const {
    if (!std::is_floating_point<T>::value) {
      diskann::cerr << "ERROR: Inner Product only defined for float currently."
                    << std::endl;
      throw diskann::ANNException(
          "ERROR: Inner Product only defined for float currently.", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }

    float result = 0;

#ifdef __GNUC__
#if defined(USE_AVX2) && (defined(__x86_64__) || defined(_WINDOWS))
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                \
  tmp2 = _mm256_loadu_ps(addr2);                \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);             \
  dest = _mm256_add_ps(dest, tmp1);

    __m256       sum;
    __m256       l0, l1;
    __m256       r0, r1;
    unsigned     D = (size + 7) & ~7U;
    unsigned     DR = D % 16;
    unsigned     DD = D - DR;
    const float *l = (float *) a;
    const float *r = (float *) b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR) {
      AVX_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      AVX_DOT(l, r, sum, l0, r0);
      AVX_DOT(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
             unpack[5] + unpack[6] + unpack[7];

#else
#if defined(__SSE2__) && (defined(__x86_64__) || defined(_WINDOWS))
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm128_loadu_ps(addr1);                \
  tmp2 = _mm128_loadu_ps(addr2);                \
  tmp1 = _mm128_mul_ps(tmp1, tmp2);             \
  dest = _mm128_add_ps(dest, tmp1);
    __m128       sum;
    __m128       l0, l1, l2, l3;
    __m128       r0, r1, r2, r3;
    unsigned     D = (size + 3) & ~3U;
    unsigned     DR = D % 16;
    unsigned     DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float        unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR) {
      case 12:
        SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
      case 8:
        SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
      case 4:
        SSE_DOT(e_l, e_r, sum, l0, r0);
      default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      SSE_DOT(l, r, sum, l0, r0);
      SSE_DOT(l + 4, r + 4, sum, l1, r1);
      SSE_DOT(l + 8, r + 8, sum, l2, r2);
      SSE_DOT(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

    float        dot0, dot1, dot2, dot3;
    const float *last = (float *) a + size;
    const float *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while ((float *) a < unroll_group) {
      dot0 = a[0] * b[0];
      dot1 = a[1] * b[1];
      dot2 = a[2] * b[2];
      dot3 = a[3] * b[3];
      result += dot0 + dot1 + dot2 + dot3;
      a += 4;
      b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while ((float *) a < last) {
      result += *a++ * *b++;
    }
#endif
#endif
#endif
    return result;
  }

  template<typename T>
  float DistanceFastL2<T>::compare(const T *a, const T *b, float norm,
                                   unsigned size) const {
    float result = -2 * DistanceInnerProduct<T>::inner_product(a, b, size);
    result += norm;
    return result;
  }

  template<typename T>
  float DistanceFastL2<T>::norm(const T *a, unsigned size) const {
    if (!std::is_floating_point<T>::value) {
      diskann::cerr << "ERROR: FastL2 only defined for float currently."
                    << std::endl;
      throw diskann::ANNException(
          "ERROR: FastL2 only defined for float currently.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }
    float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_L2NORM(addr, dest, tmp) \
  tmp = _mm256_loadu_ps(addr);      \
  tmp = _mm256_mul_ps(tmp, tmp);    \
  dest = _mm256_add_ps(dest, tmp);

    __m256       sum;
    __m256       l0, l1;
    unsigned     D = (size + 7) & ~7U;
    unsigned     DR = D % 16;
    unsigned     DD = D - DR;
    const float *l = (float *) a;
    const float *e_l = l + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR) {
      AVX_L2NORM(e_l, sum, l0);
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16) {
      AVX_L2NORM(l, sum, l0);
      AVX_L2NORM(l + 8, sum, l1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
             unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
#define SSE_L2NORM(addr, dest, tmp) \
  tmp = _mm128_loadu_ps(addr);      \
  tmp = _mm128_mul_ps(tmp, tmp);    \
  dest = _mm128_add_ps(dest, tmp);

    __m128       sum;
    __m128       l0, l1, l2, l3;
    unsigned     D = (size + 3) & ~3U;
    unsigned     DR = D % 16;
    unsigned     DD = D - DR;
    const float *l = a;
    const float *e_l = l + DD;
    float        unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR) {
      case 12:
        SSE_L2NORM(e_l + 8, sum, l2);
      case 8:
        SSE_L2NORM(e_l + 4, sum, l1);
      case 4:
        SSE_L2NORM(e_l, sum, l0);
      default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16) {
      SSE_L2NORM(l, sum, l0);
      SSE_L2NORM(l + 4, sum, l1);
      SSE_L2NORM(l + 8, sum, l2);
      SSE_L2NORM(l + 12, sum, l3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
    float        dot0, dot1, dot2, dot3;
    const float *last = (float *) a + size;
    const float *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while ((float *) a < unroll_group) {
      dot0 = a[0] * a[0];
      dot1 = a[1] * a[1];
      dot2 = a[2] * a[2];
      dot3 = a[3] * a[3];
      result += dot0 + dot1 + dot2 + dot3;
      a += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while ((float *) a < last) {
      result += (*a) * (*a);
      a++;
    }
#endif
#endif
#endif
    return result;
  }

  float AVXDistanceInnerProductFloat::compare(const float *a, const float *b,
                                              uint32_t size) const {
    float result = 0.0f;
#if defined(__x86_64__) || defined(_WINDOWS)
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                \
  tmp2 = _mm256_loadu_ps(addr2);                \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);             \
  dest = _mm256_add_ps(dest, tmp1);

    __m256       sum;
    __m256       l0, l1;
    __m256       r0, r1;
    unsigned     D = (size + 7) & ~7U;
    unsigned     DR = D % 16;
    unsigned     DD = D - DR;
    const float *l = (float *) a;
    const float *r = (float *) b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
#ifndef _WINDOWS
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
#else
    __declspec(align(32)) float unpack[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

    sum = _mm256_loadu_ps(unpack);
    if (DR) {
      AVX_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      AVX_DOT(l, r, sum, l0, r0);
      AVX_DOT(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
             unpack[5] + unpack[6] + unpack[7];
#else
    float dot0, dot1, dot2, dot3;
    const float *last = a + size;
    const float *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group) {
      dot0 = a[0] * a[0];
      dot1 = a[1] * a[1];
      dot2 = a[2] * a[2];
      dot3 = a[3] * a[3];
      result += dot0 + dot1 + dot2 + dot3;
      a += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last) {
      result += (*a) * (*a);
      a++;
    }
#endif
    return -result;
  }

  template DISKANN_DLLEXPORT class DistanceInnerProduct<float>;
  template DISKANN_DLLEXPORT class DistanceInnerProduct<int8_t>;
  template DISKANN_DLLEXPORT class DistanceInnerProduct<uint8_t>;

  template DISKANN_DLLEXPORT class DistanceFastL2<float>;
  template DISKANN_DLLEXPORT class DistanceFastL2<int8_t>;
  template DISKANN_DLLEXPORT class DistanceFastL2<uint8_t>;

}  // namespace diskann
