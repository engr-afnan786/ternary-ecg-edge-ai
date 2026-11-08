#include "ecg_integer.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/* Read whitespace-separated beats on stdin. Each output row is N score,
   V score (both Q12), and class index. Reject malformed/incomplete beats. */
int main(void) {
    int32_t beat[180], scores[2];
    char token[64];
    unsigned used = 0;
    while (scanf("%63s",token) == 1) {
        char *end;
        errno = 0;
        intmax_t v = strtoimax(token,&end,10);
        if (errno || *end || v < INT32_MIN || v > INT32_MAX) return 2;
        beat[used++] = (int32_t)v;
        if (used == 180) {
            int label = ecg_predict_uv(beat,scores);
            printf("%" PRId32 " %" PRId32 " %d\n",scores[0],scores[1],label);
            used = 0;
        }
    }
    return (used || ferror(stdin) || ferror(stdout)) ? 2 : 0;
}
