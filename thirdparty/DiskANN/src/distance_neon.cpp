#include "distance_neon.h"
#if defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <math.h>
namespace diskann {
  float NeonDistanceL2Int8::compare(const int8_t *x, const int8_t *y,
                                    uint32_t d) const {
#if defined(__ARM_NEON__) || defined(__aarch64__)
    auto      dim = d;
    int32x4_t result = {0, 0, 0, 0};
    while (d >= 16) {
      int8x8x2_t a = vld2_s8(x + dim - d);
      int8x8x2_t b = vld2_s8(y + dim - d);

      int16x8x2_t c0;
      c0.val[0] = vsubl_s8(a.val[0], b.val[0]);
      c0.val[1] = vsubl_s8(a.val[1], b.val[1]);

      int16x4x4_t c1;
      c1.val[0] = vget_high_s16(c0.val[0]);
      c1.val[1] = vget_high_s16(c0.val[1]);
      c1.val[2] = vget_low_s16(c0.val[0]);
      c1.val[3] = vget_low_s16(c0.val[1]);

      int32x4x2_t acc_sum;
      acc_sum.val[0] = vmull_s16(c1.val[0], c1.val[0]);
      acc_sum.val[1] = vmull_s16(c1.val[1], c1.val[1]);
      acc_sum.val[0] =
          vaddq_s32(acc_sum.val[0], vmull_s16(c1.val[2], c1.val[2]));
      acc_sum.val[1] =
          vaddq_s32(acc_sum.val[1], vmull_s16(c1.val[3], c1.val[3]));
      acc_sum.val[0] = vaddq_s32(acc_sum.val[0], acc_sum.val[1]);
      result = vaddq_s32(result, acc_sum.val[0]);
      d -= 16;
    }
    auto small_dim_cal = [&](int8x8_t a, int8x8_t b) {
      int16x8_t   c0 = vsubl_s8(a, b);
      int16x4x2_t c1 = {vget_high_s16(c0), vget_low_s16(c0)};
      int32x4_t   acc_sum = vmull_s16(c1.val[0], c1.val[0]);
      result = vaddq_s32(result, vmull_s16(c1.val[1], c1.val[1]));
      result = vaddq_s32(result, acc_sum);
    };

    if (d >= 8) {
      int8x8_t a = vld1_s8(x + dim - d);
      int8x8_t b = vld1_s8(y + dim - d);
      small_dim_cal(a, b);
      d -= 8;
    }

    if (d != 0) {
      int8x8_t res_x = {0, 0, 0, 0, 0, 0, 0, 0};
      int8x8_t res_y = {0, 0, 0, 0, 0, 0, 0, 0};
      switch (d) {
        case 7: {
          res_x = vld1_lane_s8(x + dim - 7, res_x, 7);
          res_y = vld1_lane_s8(y + dim - 7, res_y, 7);
        }
        case 6: {
          res_x = vld1_lane_s8(x + dim - 6, res_x, 6);
          res_y = vld1_lane_s8(y + dim - 6, res_y, 6);
        }
        case 5: {
          res_x = vld1_lane_s8(x + dim - 5, res_x, 5);
          res_y = vld1_lane_s8(y + dim - 5, res_y, 5);
        }
        case 4: {
          res_x = vld1_lane_s8(x + dim - 4, res_x, 4);
          res_y = vld1_lane_s8(y + dim - 4, res_y, 4);
        }
        case 3: {
          res_x = vld1_lane_s8(x + dim - 3, res_x, 3);
          res_y = vld1_lane_s8(y + dim - 3, res_y, 3);
        }
        case 2: {
          res_x = vld1_lane_s8(x + dim - 2, res_x, 2);
          res_y = vld1_lane_s8(y + dim - 2, res_y, 2);
        }
        case 1: {
          res_x = vld1_lane_s8(x + dim - 1, res_x, 1);
          res_y = vld1_lane_s8(y + dim - 1, res_y, 1);
        }
        default:;
      }
      small_dim_cal(res_x, res_y);
    }
    return (float) vaddlvq_s32(result);
#else
    int32_t result = 0.0f;
    for (int32_t i = 0; i < (int32_t) d; i++) {
      result += ((int32_t) ((int16_t) x[i] - (int16_t) y[i])) *
                ((int32_t) ((int16_t) x[i] - (int16_t) y[i]));
    }
    return (float) result;
#endif
  };

  float NeonDistanceL2Float::compare(const float *x, const float *y,
                                     uint32_t d) const {
#if defined(__ARM_NEON__) || defined(__aarch64__)
    float32x4_t sum_ = {0.0f, 0.0f, 0.0f, 0.0f};

    auto dim = d;
    while (d >= 16) {
      float32x4x4_t a = vld1q_f32_x4(x + dim - d);
      float32x4x4_t b = vld1q_f32_x4(y + dim - d);
      float32x4x4_t c;

      c.val[0] = vsubq_f32(a.val[0], b.val[0]);
      c.val[1] = vsubq_f32(a.val[1], b.val[1]);
      c.val[2] = vsubq_f32(a.val[2], b.val[2]);
      c.val[3] = vsubq_f32(a.val[3], b.val[3]);

      c.val[0] = vmulq_f32(c.val[0], c.val[0]);
      c.val[1] = vmulq_f32(c.val[1], c.val[1]);
      c.val[2] = vmulq_f32(c.val[2], c.val[2]);
      c.val[3] = vmulq_f32(c.val[3], c.val[3]);

      c.val[0] = vaddq_f32(c.val[0], c.val[1]);
      c.val[2] = vaddq_f32(c.val[2], c.val[3]);
      c.val[0] = vaddq_f32(c.val[0], c.val[2]);

      sum_ = vaddq_f32(sum_, c.val[0]);

      d -= 16;
    }

    if (d >= 8) {
      float32x4x2_t a = vld1q_f32_x2(x + dim - d);
      float32x4x2_t b = vld1q_f32_x2(y + dim - d);
      float32x4x2_t c;
      c.val[0] = vsubq_f32(a.val[0], b.val[0]);
      c.val[1] = vsubq_f32(a.val[1], b.val[1]);

      c.val[0] = vmulq_f32(c.val[0], c.val[0]);
      c.val[1] = vmulq_f32(c.val[1], c.val[1]);

      c.val[0] = vaddq_f32(c.val[0], c.val[1]);
      sum_ = vaddq_f32(sum_, c.val[0]);
      d -= 8;
    }
    if (d >= 4) {
      float32x4_t a = vld1q_f32(x + dim - d);
      float32x4_t b = vld1q_f32(y + dim - d);
      float32x4_t c;
      c = vsubq_f32(a, b);
      c = vmulq_f32(c, c);

      sum_ = vaddq_f32(sum_, c);
      d -= 4;
    }

    float32x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    float32x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
    if (d >= 3) {
      res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
      res_y = vld1q_lane_f32(y + dim - d, res_y, 2);
      d -= 1;
    }

    if (d >= 2) {
      res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
      res_y = vld1q_lane_f32(y + dim - d, res_y, 1);
      d -= 1;
    }

    if (d >= 1) {
      res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
      res_y = vld1q_lane_f32(y + dim - d, res_y, 0);
      d -= 1;
    }

    sum_ = vaddq_f32(
        sum_, vmulq_f32(vsubq_f32(res_x, res_y), vsubq_f32(res_x, res_y)));

    return vaddvq_f32(sum_);
#else
    float result = 0.0f;
    for (int32_t i = 0; i < (int32_t) d; i++) {
      result += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return result;
#endif
  };

  float NeonDistanceL2UInt8::compare(const uint8_t *x, const uint8_t *y,
                                     uint32_t d) const {
#if defined(__ARM_NEON__) || defined(__aarch64__)
    auto      dim = d;
    int32x4_t result = {0, 0, 0, 0};
    uint8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};
    auto      eight_dim_cal = [&](uint8x8_t s8a, uint8x8_t s8b) {
      uint8x8x2_t a = vzip_u8(vld1_u8(x + dim - d), zero);
      uint8x8x2_t b = vzip_u8(vld1_u8(y + dim - d), zero);
      int16x4x2_t s16a = {vreinterpret_s16_u8(a.val[0]),
                          vreinterpret_s16_u8(a.val[1])};
      int16x4x2_t s16b = {vreinterpret_s16_u8(b.val[0]),
                          vreinterpret_s16_u8(b.val[1])};
      s16a.val[0] = vsub_s16(s16a.val[0], s16b.val[0]);
      s16a.val[1] = vsub_s16(s16a.val[1], s16b.val[1]);
      int32x4_t acc_sum;
      acc_sum = vmull_s16(s16a.val[0], s16a.val[0]);
      result = vaddq_s32(result, vmull_s16(s16a.val[1], s16a.val[1]));
      result = vaddq_s32(result, acc_sum);
    };
    while (d >= 8) {
      uint8x8_t s8a = vld1_u8(x + dim - d);
      uint8x8_t s8b = vld1_u8(y + dim - d);
      eight_dim_cal(s8a, s8b);
      d -= 8;
    }
    if (d > 0) {
      uint8x8_t res_x = {0, 0, 0, 0, 0, 0, 0, 0};
      uint8x8_t res_y = {0, 0, 0, 0, 0, 0, 0, 0};
      switch (d) {
        case 7: {
          res_x = vld1_lane_u8(x + dim - 7, res_x, 7);
          res_y = vld1_lane_u8(y + dim - 7, res_y, 7);
        }
        case 6: {
          res_x = vld1_lane_u8(x + dim - 6, res_x, 6);
          res_y = vld1_lane_u8(y + dim - 6, res_y, 6);
        }
        case 5: {
          res_x = vld1_lane_u8(x + dim - 5, res_x, 5);
          res_y = vld1_lane_u8(y + dim - 5, res_y, 5);
        }
        case 4: {
          res_x = vld1_lane_u8(x + dim - 4, res_x, 4);
          res_y = vld1_lane_u8(y + dim - 4, res_y, 4);
        }
        case 3: {
          res_x = vld1_lane_u8(x + dim - 3, res_x, 3);
          res_y = vld1_lane_u8(y + dim - 3, res_y, 3);
        }
        case 2: {
          res_x = vld1_lane_u8(x + dim - 2, res_x, 2);
          res_y = vld1_lane_u8(y + dim - 2, res_y, 2);
        }
        case 1: {
          res_x = vld1_lane_u8(x + dim - 1, res_x, 1);
          res_y = vld1_lane_u8(y + dim - 1, res_y, 1);
        }
        default:;
      }
      eight_dim_cal(res_x, res_y);
    }

    return (float) vaddlvq_s32(result);
#else
    uint32_t result = 0.0f;
    for (int32_t i = 0; i < (int32_t) d; i++) {
      result += ((int32_t) ((int16_t) x[i] - (int16_t) y[i])) *
                ((int32_t) ((int16_t) x[i] - (int16_t) y[i]));
    }
    return (float) result;
#endif
  };

  float NeonDistanceInnerProductFloat::compare(const float *x, const float *y,
                                               uint32_t d) const {
#if defined(__ARM_NEON__) || defined(__aarch64__)
    float32x4_t sum_ = {0.0f, 0.0f, 0.0f, 0.0f};
    auto        dim = d;
    while (d >= 16) {
      float32x4x4_t a = vld1q_f32_x4(x + dim - d);
      float32x4x4_t b = vld1q_f32_x4(y + dim - d);
      float32x4x4_t c;
      c.val[0] = vmulq_f32(a.val[0], b.val[0]);
      c.val[1] = vmulq_f32(a.val[1], b.val[1]);
      c.val[2] = vmulq_f32(a.val[2], b.val[2]);
      c.val[3] = vmulq_f32(a.val[3], b.val[3]);

      c.val[0] = vaddq_f32(c.val[0], c.val[1]);
      c.val[2] = vaddq_f32(c.val[2], c.val[3]);
      c.val[0] = vaddq_f32(c.val[0], c.val[2]);

      sum_ = vaddq_f32(sum_, c.val[0]);

      d -= 16;
    }

    if (d >= 8) {
      float32x4x2_t a = vld1q_f32_x2(x + dim - d);
      float32x4x2_t b = vld1q_f32_x2(y + dim - d);
      float32x4x2_t c;
      c.val[0] = vmulq_f32(a.val[0], b.val[0]);
      c.val[1] = vmulq_f32(a.val[1], b.val[1]);
      c.val[0] = vaddq_f32(c.val[0], c.val[1]);
      sum_ = vaddq_f32(sum_, c.val[0]);
      d -= 8;
    }
    if (d >= 4) {
      float32x4_t a = vld1q_f32(x + dim - d);
      float32x4_t b = vld1q_f32(y + dim - d);
      float32x4_t c;
      c = vmulq_f32(a, b);
      sum_ = vaddq_f32(sum_, c);
      d -= 4;
    }

    float32x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    float32x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
    if (d >= 3) {
      res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
      res_y = vld1q_lane_f32(y + dim - d, res_y, 2);
      d -= 1;
    }

    if (d >= 2) {
      res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
      res_y = vld1q_lane_f32(y + dim - d, res_y, 1);
      d -= 1;
    }

    if (d >= 1) {
      res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
      res_y = vld1q_lane_f32(y + dim - d, res_y, 0);
      d -= 1;
    }

    sum_ = vaddq_f32(sum_, vmulq_f32(res_x, res_y));

    return (-1) * vaddvq_f32(sum_);
#else
    float result = 0.0f;
    for (int32_t i = 0; i < (int32_t) d; i++) {
      result += (x[i] * y[i]);
    }
    return -result;
#endif
  };

};  // namespace diskann
