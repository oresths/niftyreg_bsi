#ifndef LOCALTRANSFORMATION_H_
#define LOCALTRANSFORMATION_H_

#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <malloc.h>
#include <time.h>

#include "cudalike_math.h"

#define NUM_C 4                                 //Number of control points that affect a voxel in the 3 axes
#define NUM_THREADS    8                        //Number of threads for pthreads

struct thread_data{
   int  thread_id;
   float4_t *controlPoint;
   float4_t *positionField;
   int3_t controlPointImageDim;
   int3_t referenceImageDim;
   float3_t controlPointVoxelSpacing;
};


void reg_bspline_getDeformationFieldLerpCPU2_doubleOUT(float4_t *controlPoint, double4_t *positionField,
        const int3_t controlPointImageDim, const int3_t referenceImageDim, const float3_t controlPointVoxelSpacing);


#endif /* LOCALTRANSFORMATION_H_ */

