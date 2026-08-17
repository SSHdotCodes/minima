// Minima W1.58A8 CPU kernel.
// The I2_S layout and unsigned-code dot-product strategy are independently
// adapted from Microsoft BitNet (MIT); see repository NOTICE.

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {

inline int32_t dot_scalar(const uint8_t* packed, const int8_t* x, int group_size) {
    const int quarter = group_size / 4;
    int32_t sum = 0;
    for (int j = 0; j < quarter; ++j) {
        const uint8_t p = packed[j];
        sum += (static_cast<int>((p >> 0) & 3) - 1) * static_cast<int>(x[j]);
        sum += (static_cast<int>((p >> 2) & 3) - 1) * static_cast<int>(x[quarter + j]);
        sum += (static_cast<int>((p >> 4) & 3) - 1) * static_cast<int>(x[2 * quarter + j]);
        sum += (static_cast<int>((p >> 6) & 3) - 1) * static_cast<int>(x[3 * quarter + j]);
    }
    return sum;
}

#if defined(__AVX2__)
inline int32_t hsum256(__m256i value) {
    __m128i lo = _mm256_castsi256_si128(value);
    __m128i hi = _mm256_extracti128_si256(value, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

inline int32_t hsum128(__m128i value) {
    value = _mm_hadd_epi32(value, value);
    value = _mm_hadd_epi32(value, value);
    return _mm_cvtsi128_si32(value);
}

inline int32_t dot_avx2_32(const uint8_t* packed, const int8_t* x) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(3);
    const __m128i ones16 = _mm_set1_epi16(1);
    const __m128i ones8 = _mm_set1_epi8(1);
    __m128i acc = _mm_setzero_si128();
    __m128i xsum = _mm_setzero_si128();
    for (int lane = 0; lane < 4; ++lane) {
        const __m128i codes = _mm_and_si128(_mm_srli_epi16(bytes, 2 * lane), mask);
        const __m128i xv = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(x + lane * 8));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(codes, xv), ones16));
        xsum = _mm_add_epi32(xsum, _mm_madd_epi16(_mm_maddubs_epi16(ones8, xv), ones16));
    }
    return hsum128(_mm_sub_epi32(acc, xsum));
}

inline int32_t dot_avx2_128(const uint8_t* packed, const int8_t* x) {
    const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8 = _mm256_set1_epi8(1);
    __m256i acc = _mm256_setzero_si256();
    __m256i xsum = _mm256_setzero_si256();
    for (int lane = 0; lane < 4; ++lane) {
        const __m256i codes = _mm256_and_si256(_mm256_srli_epi16(bytes, 2 * lane), mask);
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + lane * 32));
        const __m256i pair_dot = _mm256_maddubs_epi16(codes, xv);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pair_dot, ones16));
        const __m256i pair_sum = _mm256_maddubs_epi16(ones8, xv);
        xsum = _mm256_add_epi32(xsum, _mm256_madd_epi16(pair_sum, ones16));
    }
    return hsum256(_mm256_sub_epi32(acc, xsum));
}
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
inline int32_t dot_neon_32(const uint8_t* packed, const int8_t* x) {
    const uint8x8_t bytes = vld1_u8(packed);
    const uint8x8_t mask = vdup_n_u8(3);
    const uint8x8_t one = vdup_n_u8(1);
    int32x2_t acc = vdup_n_s32(0);
    for (int lane = 0; lane < 4; ++lane) {
        const uint8x8_t codes = vand_u8(vshl_u8(bytes, vdup_n_s8(-2 * lane)), mask);
        const int8x8_t trits = vreinterpret_s8_u8(vsub_u8(codes, one));
        acc = vdot_s32(acc, trits, vld1_s8(x + lane * 8));
    }
    return vaddv_s32(acc);
}

inline int32_t dot_neon(const uint8_t* packed, const int8_t* x, int group_size) {
    const int quarter = group_size / 4;
    int32x4_t acc = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);
    const uint8x16_t one = vdupq_n_u8(1);
    int j = 0;
    for (; j + 16 <= quarter; j += 16) {
        const uint8x16_t bytes = vld1q_u8(packed + j);
        for (int lane = 0; lane < 4; ++lane) {
            const uint8x16_t codes = vandq_u8(vshlq_u8(bytes, vdupq_n_s8(-2 * lane)), mask);
            const int8x16_t trits = vreinterpretq_s8_u8(vsubq_u8(codes, one));
            const int8x16_t xv = vld1q_s8(x + lane * quarter + j);
            acc = vdotq_s32(acc, trits, xv);
        }
    }
    int32_t result = vaddvq_s32(acc);
    if (j < quarter) {
        result += dot_scalar(packed + j, x + j, (quarter - j) * 4);
    }
    return result;
}
#endif

inline int32_t dot_group(const uint8_t* packed, const int8_t* x, int group_size) {
#if defined(__AVX2__)
    if (group_size == 32) return dot_avx2_32(packed, x);
    if (group_size == 128) return dot_avx2_128(packed, x);
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (group_size == 32) return dot_neon_32(packed, x);
    if ((group_size % 64) == 0) return dot_neon(packed, x, group_size);
#endif
    return dot_scalar(packed, x, group_size);
}

}  // namespace

torch::Tensor i2s_linear(torch::Tensor input, torch::Tensor packed, torch::Tensor scales,
                         int64_t in_features, int64_t group_size) {
    TORCH_CHECK(input.device().is_cpu() && packed.device().is_cpu() && scales.device().is_cpu(),
                "i2s_linear is a CPU kernel");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed weights must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(input.dim() == 2 && packed.dim() == 3 && scales.dim() == 2, "invalid tensor rank");
    TORCH_CHECK(input.is_contiguous() && packed.is_contiguous() && scales.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(input.size(1) == in_features, "input feature mismatch");
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0, "invalid group size");

    const int64_t m = input.size(0);
    const int64_t n = packed.size(0);
    const int64_t groups = packed.size(1);
    const int64_t padded_k = groups * group_size;
    TORCH_CHECK(packed.size(2) == group_size / 4, "invalid I2_S quarter width");
    TORCH_CHECK(scales.size(0) == n && scales.size(1) == groups, "scale shape mismatch");

    auto qx = torch::empty({m, padded_k}, torch::TensorOptions().dtype(torch::kInt8));
    auto xscales = torch::empty({m, groups}, torch::TensorOptions().dtype(torch::kFloat32));
    const float* xptr = input.data_ptr<float>();
    int8_t* qptr = qx.data_ptr<int8_t>();
    float* xsptr = xscales.data_ptr<float>();

    at::parallel_for(0, m * groups, 1, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t row = index / groups;
            const int64_t group = index % groups;
            const int64_t start = group * group_size;
            const int64_t valid = std::max<int64_t>(0, std::min<int64_t>(group_size, in_features - start));
            float maxabs = 0.0f;
            for (int64_t j = 0; j < valid; ++j) maxabs = std::max(maxabs, std::abs(xptr[row * in_features + start + j]));
            const float scale = std::max(maxabs / 127.0f, 1.0e-8f);
            xsptr[index] = scale;
            int8_t* dst = qptr + row * padded_k + start;
            for (int64_t j = 0; j < valid; ++j) {
                const float value = std::nearbyint(xptr[row * in_features + start + j] / scale);
                dst[j] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, value)));
            }
            std::fill(dst + valid, dst + group_size, static_cast<int8_t>(0));
        }
    });

    auto output = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kFloat32));
    const uint8_t* wptr = packed.data_ptr<uint8_t>();
    const float* wsptr = scales.data_ptr<float>();
    float* out = output.data_ptr<float>();
    const int64_t packed_group = group_size / 4;

    at::parallel_for(0, m * n, 1, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t row = index / n;
            const int64_t col = index % n;
            float sum = 0.0f;
            for (int64_t group = 0; group < groups; ++group) {
                const uint8_t* w = wptr + (col * groups + group) * packed_group;
                const int8_t* x = qptr + row * padded_k + group * group_size;
                sum += static_cast<float>(dot_group(w, x, static_cast<int>(group_size))) *
                       xsptr[row * groups + group] * wsptr[col * groups + group];
            }
            out[index] = sum;
        }
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("i2s_linear", &i2s_linear, "Fused W1.58A8 I2_S linear (CPU)");
}
