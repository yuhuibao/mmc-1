/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504, 2018.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mmcx_core.h

@brief   MMC GPU kernel header file
*******************************************************************************/

#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include "mcx_utils.h"
#include "simpmesh.h"
#include "tettracing.h"

#ifdef  __cplusplus
extern "C" {
#endif


#define ABS(a)  ((a)<0?-(a):(a))
#define DETINC	32

#define MAX_ACCUM           1000.f

#define ROULETTE_SIZE       10.f                /**< Russian Roulette size */

#ifdef  MCX_DEBUG
#define GPUDEBUG(x)        printf x             /**< enable debugging in CPU mode */
#else
#define GPUDEBUG(x)
#endif



typedef unsigned char uchar;

/**
 * @brief Simulation constant parameters stored in the constant memory
 *
 * This struct stores all constants used in the simulation.
 */

typedef struct MMC_Parameter {
  float3 srcpos;
  float3 srcdir;
  float  tstart,tend;
  uint   isreflect,issavedet,issaveexit,ismomentum,isatomic,isspecular;
  float  Rtstep;
  float  minenergy;
  uint   maxdetphoton;
  uint   maxmedia;
  uint   detnum;
  int    voidtime;
  int    srctype;                    /**< type of the source */
  float4 srcparam1;                  /**< source parameters set 1 */
  float4 srcparam2;                  /**< source parameters set 2 */
  uint   issaveref;     /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
  uint   maxgate;
  uint   debuglevel;           /**< debug flags */
  int    reclen;                 /**< record (4-byte per record) number per detected photon */
  int    outputtype;
  int    elemlen;
  int    mcmethod;
  int    method;
  float  dstep;
  float  focus;
  int    nn, ne, nf;
  float3 nmin;
  float  nout;
  uint   roulettesize;
  int    srcnum;
  int4   crop0;
  int    srcelemlen;
  float4 bary0;
  int    e0;
  int    isextdet;
  int    framelen;
  uint   nbuffer;
  uint   buffermask;
  //int    issaveseed;
} MCXParam __attribute__ ((aligned (32)));

typedef struct GPU_reporter{
  float  raytet;
} MCXReporter  __attribute__ ((aligned (32)));

void mmc_run_cl(mcconfig *cfg, tetmesh *mesh, raytracer *tracer);

int mcx_list_gpu(mcconfig *cfg, GPUInfo **info);
#ifdef  __cplusplus
}
#endif

#endif

