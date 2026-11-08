#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <inttypes.h>
extern void bench_init(void);
extern int ecg_predict_uv(const int32_t*,int32_t*);
static uint64_t ns(void) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t)) exit(2);
    return (uint64_t)t.tv_sec*1000000000u+t.tv_nsec;
}
int main(void) {
    int32_t uv[180],scores[2];
    bench_init();
    size_t count;
    unsigned beats=0;
    while((count=fread(uv,sizeof(*uv),180,stdin))==180) {
        if(!beats) for(int j=0;j<32;j++) (void)ecg_predict_uv(uv,scores);
        uint64_t start=ns();
        int label=ecg_predict_uv(uv,scores);
        uint64_t elapsed=ns()-start;
        printf("%"PRIu64" %"PRId32" %"PRId32" %d\n",elapsed,scores[0],scores[1],label);
        beats++;
    }
    struct rusage usage;
    if(getrusage(RUSAGE_SELF,&usage)) return 2;
    /* Linux /proc gives this exec image's memory high-water mark. */
    FILE *f=fopen("/proc/self/status","r");
    char line[256]; long kib=0;
    if(f) {
        while(fgets(line,sizeof(line),f)) if(sscanf(line,"VmHWM: %ld",&kib)==1) break;
        fclose(f);
    }
    fprintf(stderr,"%ld\n",kib?kib:usage.ru_maxrss);
    return count || !beats || ferror(stdin) || ferror(stdout);
}
