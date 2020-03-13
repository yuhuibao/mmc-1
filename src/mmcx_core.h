#ifndef _MMC_CORE_H
#define _MMC_CORE_H
#include "mcx_utils.h"

typedef struct MMC_Ray{
	float3 p0;                    /**< current photon position */
	float3 vec;                   /**< current photon direction vector */
	float3 pout;                  /**< the intersection position of the ray to the enclosing tet */
	int eid;                      /**< the index of the enclosing tet (starting from 1) */
	int faceid;                   /**< the index of the face at which ray intersects with tet */
	int isend;                    /**< if 1, the scattering event ends before reaching the intersection */
	float weight;                 /**< photon current weight */
	float photontimer;            /**< the total time-of-fly of the photon */
	float slen;                   /**< the remaining unitless scattering length = length*mus  */
	float Lmove;                  /**< last photon movement length */
	uint oldidx;
	float oldweight;
	//int nexteid;                  /**< the index to the neighboring tet to be moved into */
	//float4 bary0;                 /**< the Barycentric coordinate of the intersection with the tet */
	//float slen0;                  /**< initial unitless scattering length = length*mus */
	//unsigned int photonid;        /**< index of the current photon */
	//unsigned int posidx;	      /**< launch position index of the photon for pattern source type */
} ray __attribute__ ((aligned (32)));

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
  float4 nmin;
  float  nout;
  uint   roulettesize;
  int    srcnum;
  uint3   crop0;
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


#endif
