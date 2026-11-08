#include "ecg_integer.h"
#include "integer_config.h"
#if defined(BENCH_UNPACKED)
#include "byte_weights.h"
typedef int8_t weight_byte;
#else
#include "ternary_weights.h"
typedef uint8_t weight_byte;
#endif

/* Round signed values to nearest, with ties away from zero. Build-time
   bounds in the generator guarantee room for the rounding offset. */
static int64_t rounded(int64_t value, int32_t divisor) {
    int64_t half = divisor / 2;
    return value >= 0 ? (value + half) / divisor : (value - half) / divisor;
}

/* Keep Q12 activations within the verified input range for the next layer. */
static int32_t bound(int64_t value, int32_t low) {
    if (value < low) return low;
    if (value > ECG_LIMIT) return ECG_LIMIT;
    return (int32_t)value;
}

/* Extract one signed ternary symbol, least significant digit first. */
static int symbol(const weight_byte *packed, unsigned index) {
#if defined(BENCH_UNPACKED)
    return packed[index];
#elif ECG_PACKING == 2
    return (int)((packed[index/4] >> (2*(index%4))) & 3u) - 1;
#else
    static const unsigned powers[5] = {1,3,9,27,81};
    return (int)((packed[index/5] / powers[index%5]) % 3u) - 1;
#endif
}

/* A ternary product needs only a sign choice. Zero contributes nothing. */
static int32_t term(int code, int32_t value) {
    if (code == 1) return value;
    if (code == -1) return -value;
    return 0;
}

/* Cross-correlate one output element using channel-major arrays and zero padding. */
static int32_t conv_at(const int32_t *x, int length, int channels,
                       const weight_byte *weights, int output, int start, int kernel) {
    int32_t sum = 0;
    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < kernel; ++k) {
            int t = start + k;
            if (t >= 0 && t < length) {
                unsigned index = (output*channels+c)*kernel+k;
#ifdef BENCH_CACHE
                int code = (int)weights[index]-1;
#else
                int code = symbol(weights,index);
#endif
                sum += term(code, x[c*length+t]);
            }
        }
    }
    return sum;
}

/* Convert a Q12 dot product back to Q12 using a Q20 layer scale. */
static int32_t affine(int32_t sum, int32_t scale, int32_t bias, int32_t low) {
    return bound(rounded((int64_t)sum * scale, ECG_ONE20) + bias, low);
}

/* The cached variant keeps one signed-byte digit per code in RAM after startup. */
#ifdef BENCH_CACHE
static uint8_t cache1[56], cache2[640], cache_head[32];
void bench_init(void) {
    for (unsigned i=0;i<56;i++) cache1[i]=(uint8_t)(symbol(conv1_packed,i)+1);
    for (unsigned i=0;i<640;i++) cache2[i]=(uint8_t)(symbol(conv2_packed,i)+1);
    for (unsigned i=0;i<32;i++) cache_head[i]=(uint8_t)(symbol(head_packed,i)+1);
}
#else
void bench_init(void) {}
#endif
int ecg_predict_uv(const int32_t uv[180], int32_t scores[2]) {
#ifdef BENCH_CACHE
#define conv1_packed cache1
#define conv2_packed cache2
#endif
    int32_t input[180], pooled[8*45], average[16];
    for (int t = 0; t < 180; ++t) {
        int64_t centered = (int64_t)uv[t] - ECG_MEAN_UV;
        input[t] = bound(rounded(centered * ECG_INPUT_GAIN, ECG_ONE20), -ECG_LIMIT);
    }
    for (int c = 0; c < 8; ++c) {
        for (int t = 0; t < 45; ++t) {
            int32_t a = conv_at(input,180,1,conv1_packed,c,4*t-3,7);
            int32_t b = conv_at(input,180,1,conv1_packed,c,4*t-1,7);
            /* ReLU and a nonnegative scale make max commute with requantization. */
            pooled[c*45+t] = affine(a>b?a:b, conv1_scale, conv1_bias_q[c], 0);
        }
    }
    for (int c = 0; c < 16; ++c) {
        int32_t total = 0;
        for (int t = 0; t < 45; ++t) {
            int32_t dot = conv_at(pooled,45,8,conv2_packed,c,t-2,5);
            total += affine(dot, conv2_scale, conv2_bias_q[c], 0);
        }
        average[c] = (int32_t)rounded(total,45);
    }
    for (int c = 0; c < 2; ++c) {
        int32_t dot = 0;
        for (int k = 0; k < 16; ++k)
            #ifdef BENCH_CACHE
            dot += term((int)cache_head[c*16+k]-1,average[k]);
#else
            dot += term(symbol(head_packed,c*16+k),average[k]);
#endif
        scores[c] = affine(dot,head_scale,head_bias_q[c],-ECG_LIMIT);
    }
    return scores[1] > scores[0];
}
