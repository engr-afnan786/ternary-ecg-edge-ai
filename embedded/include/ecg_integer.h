#ifndef ECG_INTEGER_H
#define ECG_INTEGER_H
#include <stdint.h>

/* Infer one 180-sample beat in integer microvolts. Write two Q12 scores.
   Return the winning index (0=N, 1=V), choosing N on a tie. */
int ecg_predict_uv(const int32_t uv[180], int32_t scores[2]);
#endif
