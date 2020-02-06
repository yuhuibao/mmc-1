#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define ABS(a)  ((a)<0?-(a):(a))

#define MCX_DEBUG_RNG       2                   /**< MCX debug flags */
#define MCX_DEBUG_MOVE      1
#define MCX_DEBUG_PROGRESS  2048

#define MIN(a,b)           ((a)<(b)?(a):(b))

#define OCL_ASSERT(x)  ocl_assess((x),__FILE__,__LINE__)


int mcx_list_gpu(mcconfig *cfg,GPUInfo **info);
void ocl_assess(int cuerr,const char *file,const int linenum);

#ifdef  __cplusplus
}
#endif

#endif
