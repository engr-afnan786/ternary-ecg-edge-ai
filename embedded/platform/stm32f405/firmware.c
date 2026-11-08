#include <stdint.h>
#include "ecg_integer.h"
#include "phase9_vectors.h"

extern int semihost(unsigned op, const void *arg);

/* Compare both signed Q12 scores and the winning class on every vector. */
static int check_vectors(void) {
    for (unsigned i = 0; i < VECTOR_COUNT; ++i) {
        int32_t scores[2];
        int label = ecg_predict_uv(inputs[i], scores);
        if (scores[0] != expected[i][0] || scores[1] != expected[i][1] ||
            label != expected[i][2]) return 1;
    }
    return 0;
}

void firmware_main(void) {
    int failed = check_vectors();
    uint32_t exit_args[2] = {0x20026u, (uint32_t)failed};
    semihost(4u, failed ? "ECG_ARM_FAIL\n" : "ECG_ARM_PASS\n");
    semihost(0x20u, exit_args);
    for (;;) {}
}
