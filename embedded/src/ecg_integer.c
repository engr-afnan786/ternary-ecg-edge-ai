#include "ecg_integer.h"
#include "integer_config.h"
#include "ternary_weights.h"

/* The verification build records intermediate values for one beat per call. */
#ifdef ECG_VERIFY_TRACE
static int32_t trace_values[1998];
const int32_t *ecg_read_trace(void) { return trace_values; }
#define TRACE(index, value) (trace_values[(index)] = (value))
#else
#define TRACE(index, value) ((void)0)
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
static int symbol(const uint8_t *packed, unsigned index) {
#if ECG_PACKING == 2
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
                       const uint8_t *weights, int output, int start, int kernel) {
    int32_t sum = 0;
    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < kernel; ++k) {
            int t = start + k;
            if (t >= 0 && t < length) {
                int code = symbol(weights, (output*channels+c)*kernel+k);
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

int ecg_predict_uv(const int32_t uv[180], int32_t scores[2]) {
    int32_t input[180], pooled[8*45], average[16];
    for (int t = 0; t < 180; ++t) {
        int64_t centered = (int64_t)uv[t] - ECG_MEAN_UV;
        input[t] = bound(rounded(centered * ECG_INPUT_GAIN, ECG_ONE20), -ECG_LIMIT);
        TRACE(t, input[t]);
    }
    for (int c = 0; c < 8; ++c) {
        for (int t = 0; t < 45; ++t) {
            int32_t a = conv_at(input,180,1,conv1_packed,c,4*t-3,7);
            int32_t b = conv_at(input,180,1,conv1_packed,c,4*t-1,7);
            TRACE(180+c*90+2*t, affine(a,conv1_scale,conv1_bias_q[c],0));
            TRACE(180+c*90+2*t+1, affine(b,conv1_scale,conv1_bias_q[c],0));
            /* ReLU and a nonnegative scale make max commute with requantization. */
            pooled[c*45+t] = affine(a>b?a:b, conv1_scale, conv1_bias_q[c], 0);
            TRACE(900+c*45+t, pooled[c*45+t]);
        }
    }
    for (int c = 0; c < 16; ++c) {
        int32_t total = 0;
        for (int t = 0; t < 45; ++t) {
            int32_t dot = conv_at(pooled,45,8,conv2_packed,c,t-2,5);
            int32_t activated = affine(dot, conv2_scale, conv2_bias_q[c], 0);
            total += activated;
            TRACE(1260+c*45+t, activated);
        }
        average[c] = (int32_t)rounded(total,45);
        TRACE(1980+c, average[c]);
    }
    for (int c = 0; c < 2; ++c) {
        int32_t dot = 0;
        for (int k = 0; k < 16; ++k)
            dot += term(symbol(head_packed,c*16+k),average[k]);
        scores[c] = affine(dot,head_scale,head_bias_q[c],-ECG_LIMIT);
        TRACE(1996+c, scores[c]);
    }
    return scores[1] > scores[0];
}
