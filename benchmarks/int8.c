/* Symmetric signed INT8 operands and INT32 dot products. Use FP32 scales
   for requantization and dequantization. One scale per tensor. */
#include <math.h>
#include <stdint.h>
#include "baseline8.h"
void bench_init(void) {}
static int8_t quant(float value) {
    return (int8_t)lrintf(fmaxf(-127.f,fminf(127.f,value)));
}
int ecg_predict_uv(const int32_t uv[180],int32_t scores[2]) {
    int8_t x[180],pool[8][45],avg[16];
    for(int t=0;t<180;t++) x[t]=quant(((uv[t]*0.001f-mean_mv)/std_mv)/s0);
    for(int c=0;c<8;c++) for(int t=0;t<45;t++) {
        int32_t best=0;
        for(int j=0;j<2;j++) {
            int32_t dot=0;
            for(int k=0;k<7;k++) {
                int pos=4*t+2*j+k-3;
                if(pos>=0 && pos<180) dot+=x[pos]*w1[c*7+k];
            }
            int8_t a=quant((dot*(s0*sw1)+b1[c])/s1);
            if(a>best) best=a;
        }
        pool[c][t]=(int8_t)best;
    }
    for(int c=0;c<16;c++) {
        int total=0;
        for(int t=0;t<45;t++) {
            int32_t dot=0;
            for(int i=0;i<8;i++) for(int k=0;k<5;k++) {
                int pos=t+k-2;
                if(pos>=0 && pos<45) dot+=pool[i][pos]*w2[(c*8+i)*5+k];
            }
            int8_t a=quant((dot*(s1*sw2)+b2[c])/s2);
            if(a>0) total+=a;
        }
        avg[c]=quant(total/45.f);
    }
    float logits[2];
    for(int c=0;c<2;c++) {
        int32_t dot=0;
        for(int i=0;i<16;i++) dot+=avg[i]*w3[c*16+i];
        logits[c]=dot*(s2*sw3)+b3[c];
        scores[c]=(int32_t)lrintf(logits[c]*4096.f);
    }
    return logits[1]>logits[0];
}
