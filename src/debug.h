#ifndef DEBUG_H_
#define DEBUG_H_

#ifdef MCX_GPU_DEBUG
#define MMC_PRINT(x) printf x  // enable debugging in CPU mode
#else
#define MMC_PRINT(x) \
  {}
#endif

#ifdef MCX_CONTAINER
#ifdef _OPENMP
#define MMC_FPRINTF(fp, ...)                               \
  {                                                        \
    if (omp_get_thread_num() == 0) mexPrintf(__VA_ARGS__); \
  }
#else
#define MMC_FPRINTF(fp, ...) mexPrintf(__VA_ARGS__)
#endif
#else
#define MMC_FPRINTF(fp, ...) fprintf(fp, __VA_ARGS__)
#define MCX_FPRINTF(fp, ...) fprintf(fp, __VA_ARGS__)
// #define MMC_FPRINTF(x) printf x
#endif

#endif  // DEBUG_H_
