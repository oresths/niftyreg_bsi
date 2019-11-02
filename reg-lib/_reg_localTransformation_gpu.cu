/*
 *  _reg_bspline_gpu.cu
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_GPU_CU
#define _REG_BSPLINE_GPU_CU

#include "_reg_localTransformation_gpu.h"
#include "_reg_localTransformation_kernels.cu"

#include "localTransformation.h"

#include <stdio.h>

/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_gpu(nifti_image *controlPointImage,
                     nifti_image *reference,
                     float4 **controlPointImageArray_d,
                     float4 **positionFieldImageArray_d,
                     int **mask_d,
                     int activeVoxelNumber,
                     bool bspline)
{
    const int voxelNumber = reference->nx * reference->ny * reference->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(reference->nx, reference->ny, reference->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int useBSpline = bspline;

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / reference->dx,
        controlPointImage->dy / reference->dy,
        controlPointImage->dz / reference->dz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_UseBSpline,&useBSpline,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)))

    const unsigned int Grid_reg_bspline_getDeformationField =
        (unsigned int)ceilf(sqrtf((float)activeVoxelNumber/(float)(Block_reg_bspline_getDeformationField)));
    dim3 G1(Grid_reg_bspline_getDeformationField,Grid_reg_bspline_getDeformationField,1);
    dim3 B1(Block_reg_bspline_getDeformationField,1,1);
	reg_bspline_getDeformationField0 <<< G1, B1 >>>(*positionFieldImageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
    return;
}
/* *************************************************************** */
/* ************* Here we evaluate the performance of the GPU B-spline interpolation implementations ************ */
void reg_bspline_testing(nifti_image *controlPointImage,
                     nifti_image *reference,
                     float4 **controlPointImageArray_d,
                     float4 **positionFieldImageArray_d,
                     int **mask_d,
                     int activeVoxelNumber,
                     bool bspline)
{
    const int voxelNumber = reference->nx * reference->ny * reference->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(reference->nx, reference->ny, reference->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int useBSpline = bspline;

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / reference->dx,
        controlPointImage->dy / reference->dy,
        controlPointImage->dz / reference->dz);
    const int3 controlPointVoxelSpacingInt = make_int3(
        lrintf( controlPointImage->dx / reference->dx ),
        lrintf( controlPointImage->dy / reference->dy ),
        lrintf( controlPointImage->dz / reference->dz ));

    const int3_t referenceImageDimCPU = make_int3_t(reference->nx, reference->ny, reference->nz);
    const int3_t controlPointImageDimCPU = make_int3_t(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);

    const float3_t controlPointVoxelSpacingCPU = make_float3_t(
        controlPointImage->dx / reference->dx,
        controlPointImage->dy / reference->dy,
        controlPointImage->dz / reference->dz);

    //Doesn't affect a lot, but I/O is preallocated to avoid page faults
    float4_t * positionFieldImageArray_h = (float4_t *)calloc(
            referenceImageDimCPU.x*referenceImageDimCPU.y*referenceImageDimCPU.z , sizeof(float4_t));

    float4_t *controlPointImageArray_h=(float4_t *)malloc(controlPointNumber * sizeof(float4_t));

    const size_t controlCount(controlPointNumber);
    const size_t controlMemory(controlCount * sizeof(float4_t));
    float4_t* controlTemp = new float4_t[controlCount];

    NR_CUDA_SAFE_CALL(cudaMemcpy(controlTemp, *controlPointImageArray_d, controlMemory, cudaMemcpyDeviceToHost))

    for(int i=0; i<controlPointNumber; i++){
        controlPointImageArray_h[i] = controlTemp[i];
    }

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_UseBSpline,&useBSpline,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_controlPointVoxelSpacingInt,&controlPointVoxelSpacingInt,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)))

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t controlPoints3D;
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(&controlPoints3D, &channelDesc,
            make_cudaExtent(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz)))

    cudaMemcpy3DParms pars3D = {0};
    pars3D.dstArray = controlPoints3D;
    cudaExtent ext = make_cudaExtent(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    pars3D.extent = ext;
    pars3D.kind = cudaMemcpyDeviceToDevice;
    pars3D.srcPtr = make_cudaPitchedPtr(*controlPointImageArray_d, controlPointImage->nx*sizeof(float4),
            controlPointImage->nx, controlPointImage->ny);

    NR_CUDA_SAFE_CALL( cudaMemcpy3D(&pars3D) )

    controlPoints3Dtex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(&controlPoints3Dtex, controlPoints3D, &channelDesc);

    const int3 tilesDim = make_int3(ceilf((float)reference->nx / (float)controlPointVoxelSpacing.x),
                                    ceilf((float)reference->ny / (float)controlPointVoxelSpacing.y),
                                    ceilf((float)reference->nz / (float)controlPointVoxelSpacing.z));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_tilesDim,&tilesDim,sizeof(int3)))

    const float3 tilesDim_f = make_float3((float)reference->nx / (float)controlPointVoxelSpacing.x,
                                          (float)reference->ny / (float)controlPointVoxelSpacing.y,
                                          (float)reference->nz / (float)controlPointVoxelSpacing.z);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_tilesDim_f,&tilesDim_f,sizeof(float3)))

    float x_g0[MAX_CURRENT_SPACE];
    float x_h0[MAX_CURRENT_SPACE];
    float x_h1[MAX_CURRENT_SPACE];
    float y_g0[MAX_CURRENT_SPACE];
    float y_h0[MAX_CURRENT_SPACE];
    float y_h1[MAX_CURRENT_SPACE];
    float z_g0[MAX_CURRENT_SPACE];
    float z_h0[MAX_CURRENT_SPACE];
    float z_h1[MAX_CURRENT_SPACE];
    float x_h0_r[MAX_CURRENT_SPACE];
    float x_h1_r[MAX_CURRENT_SPACE];
    float y_h0_r[MAX_CURRENT_SPACE];
    float y_h1_r[MAX_CURRENT_SPACE];
    float z_h0_r[MAX_CURRENT_SPACE];
    float z_h1_r[MAX_CURRENT_SPACE];
    float x_h01_r[MAX_CURRENT_SPACE*2];
    float xBasis[NUM_C*MAX_CURRENT_SPACE];
    float yBasis[NUM_C*MAX_CURRENT_SPACE];
    float zBasis[NUM_C*MAX_CURRENT_SPACE];
    float relative;
    for (int i = 0; i < controlPointVoxelSpacingInt.x; ++i) {
        relative = (float) i / controlPointVoxelSpacing.x;
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.f-relative;
        xBasis[0+NUM_C*i] = (MF)*(MF)*(MF)/(6.f);
        xBasis[1+NUM_C*i] = (3.f*FFF - 6.f*FF + 4.f)/6.f;
        xBasis[2+NUM_C*i] = (-3.f*FFF + 3.f*FF + 3.f*relative + 1.f)/6.f;
        xBasis[3+NUM_C*i] = (FFF/6.f);

        x_g0[i] = xBasis[0+NUM_C*i] + xBasis[1+NUM_C*i];
        x_h0[i] = xBasis[1+NUM_C*i] / (x_g0[i]) - 1;
        x_h1[i] = xBasis[3+NUM_C*i] / (1 - x_g0[i]) + 1;

        x_h0_r[i] = xBasis[1+NUM_C*i] / (x_g0[i]);
        x_h1_r[i] = xBasis[3+NUM_C*i] / (1 - x_g0[i]);

        x_h01_r[i] = x_h0_r[i];
        x_h01_r[i+(int)controlPointVoxelSpacingInt.x] = x_h1_r[i];
    }
    for (int i = 0; i < controlPointVoxelSpacingInt.y; ++i) {
        relative = (float) i / controlPointVoxelSpacing.y;
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.f-relative;
        yBasis[0+NUM_C*i] = (MF)*(MF)*(MF)/(6.f);
        yBasis[1+NUM_C*i] = (3.f*FFF - 6.f*FF + 4.f)/6.f;
        yBasis[2+NUM_C*i] = (-3.f*FFF + 3.f*FF + 3.f*relative + 1.f)/6.f;
        yBasis[3+NUM_C*i] = (FFF/6.f);

        y_g0[i] = yBasis[0+NUM_C*i] + yBasis[1+NUM_C*i];
        y_h0[i] = yBasis[1+NUM_C*i] / (y_g0[i]) - 1;
        y_h1[i] = yBasis[3+NUM_C*i] / (1 - y_g0[i]) + 1;

        y_h0_r[i] = yBasis[1+NUM_C*i] / (y_g0[i]);
        y_h1_r[i] = yBasis[3+NUM_C*i] / (1 - y_g0[i]);
    }
    for (int i = 0; i < controlPointVoxelSpacingInt.z; ++i) {
        relative = (float) i / controlPointVoxelSpacing.z;
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.f-relative;
        zBasis[0+NUM_C*i] = (MF)*(MF)*(MF)/(6.f);
        zBasis[1+NUM_C*i] = (3.f*FFF - 6.f*FF + 4.f)/6.f;
        zBasis[2+NUM_C*i] = (-3.f*FFF + 3.f*FF + 3.f*relative + 1.f)/6.f;
        zBasis[3+NUM_C*i] = (FFF/6.f);

        z_g0[i] = zBasis[0+NUM_C*i] + zBasis[1+NUM_C*i];
        z_h0[i] = zBasis[1+NUM_C*i] / (z_g0[i]) - 1;
        z_h1[i] = zBasis[3+NUM_C*i] / (1 - z_g0[i]) + 1;

        z_h0_r[i] = zBasis[1+NUM_C*i] / (z_g0[i]);
        z_h1_r[i] = zBasis[3+NUM_C*i] / (1 - z_g0[i]);
    }

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_xBasis,&xBasis,NUM_C*MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_yBasis,&yBasis,NUM_C*MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_zBasis,&zBasis,NUM_C*MAX_CURRENT_SPACE*sizeof(float)))

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_g0,&x_g0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_h0,&x_h0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_h1,&x_h1,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_g0,&y_g0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_h0,&y_h0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_h1,&y_h1,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_g0,&z_g0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_h0,&z_h0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_h1,&z_h1,MAX_CURRENT_SPACE*sizeof(float)))

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_h0_r,&x_h0_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_h1_r,&x_h1_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_h0_r,&y_h0_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_h1_r,&y_h1_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_h0_r,&z_h0_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_h1_r,&z_h1_r,MAX_CURRENT_SPACE*sizeof(float)))

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_h01_r,&x_h01_r,MAX_CURRENT_SPACE*2*sizeof(float)))

    const unsigned int Grid_reg_bspline_getDeformationField0 =
        (unsigned int)ceilf(sqrtf((float)activeVoxelNumber/(float)(Block_reg_bspline_getDeformationField)));
    dim3 G0(Grid_reg_bspline_getDeformationField0,Grid_reg_bspline_getDeformationField0,1);
    dim3 B0(256,1,1);

    const dim3 BD(8, 8, 4);
    const dim3 GD(ceilf((float)referenceImageDim.x / BD.x), ceilf((float)referenceImageDim.y / BD.y),
            ceilf((float)referenceImageDim.z / BD.z));

    unsigned int block_size = 128;
    const unsigned int Grid_reg_bspline_getDeformationField =
            (unsigned int) ceilf((float)(tilesDim.x * tilesDim.y * tilesDim.z) / block_size);

    dim3 G1(Grid_reg_bspline_getDeformationField,1,1);
    dim3 B1(block_size,1,1);

    dim3 B2(BLOCK_BS_X,BLOCK_BS_Y,BLOCK_BS_Z);
    dim3 G2((unsigned int) ceilf((float)(tilesDim.x) / B2.x),
            (unsigned int) ceilf((float)(tilesDim.y) / B2.y),
            (unsigned int) ceilf((float)(tilesDim.z) / B2.z));

    dim3 G3(tilesDim.x, tilesDim.y, tilesDim.z);

    dim3 B5((int)(ceilf((float)(controlPointVoxelSpacing.y * controlPointVoxelSpacing.z) / 2.f))
            * controlPointVoxelSpacing.x,1,1);
    dim3 B6((int)(ceilf((float)(controlPointVoxelSpacing.y * controlPointVoxelSpacing.z) / 4.f))
            * controlPointVoxelSpacing.x,1,1);

    FILE *f = fopen("results.txt", "a"); //id, level, data_in, execution, data_out, accuracy, errors
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    static int level;
    static int currentVoxelNumber = 0;
    if (currentVoxelNumber == 0) {
        currentVoxelNumber = activeVoxelNumber;
        level = 0;
    } else if (activeVoxelNumber > currentVoxelNumber) {
        currentVoxelNumber = activeVoxelNumber;
        level++;
    }


#define ID 3 //findme, valid values = 1,2,3,4,5

#if ID == 0

#elif ID
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t input3D;
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(&input3D, &channelDescT, make_cudaExtent(controlPointImageDim.x, controlPointImageDim.y, controlPointImageDim.z)))

    cudaMemcpy3DParms pars3D_T = {0};
    pars3D_T.dstArray = input3D;
    cudaExtent ext_T = make_cudaExtent(controlPointImageDim.x, controlPointImageDim.y, controlPointImageDim.z);
    pars3D_T.extent = ext_T;
    pars3D_T.kind = cudaMemcpyHostToDevice;
    pars3D_T.srcPtr = make_cudaPitchedPtr(controlPointImageArray_h, controlPointImageDim.x*sizeof(float4), controlPointImageDim.x, controlPointImageDim.y);

    bool executed = true;

    cudaEventRecord(start);

    NR_CUDA_SAFE_CALL( cudaMemcpy3D(&pars3D_T) )

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_memin = 0;
    cudaEventElapsedTime(&ms_memin, start, stop);

    float4* preheat;
    const dim3 BP(8, 8, 4);
    const dim3 GP(30, 30, 30);
    NR_CUDA_SAFE_CALL(cudaMalloc((void**)&preheat, BP.x*BP.y*BP.z*sizeof(float4)))
    for (int i = 0; i < 20000; ++i) {
        warmup <<< GP, BP >>>(preheat);
    }

    cudaEventRecord(start);

#if ID == 1
    reg_bspline_getDeformationFieldTile7_noSh<<<G2, B2 >>>(*positionFieldImageArray_d, *controlPointImageArray_d);
#endif
#if ID == 2
    reg_bspline_getDeformationFieldDanny <<< GD, BD >>>(*positionFieldImageArray_d);
#endif
#if ID == 3
    reg_bspline_getDeformationFieldTileLerp3_noSh<<<G2, B2 >>>(*positionFieldImageArray_d, *controlPointImageArray_d);
#endif
#if ID == 4
    if (B5.x <= 32)
        reg_bspline_getDeformationFieldTileVoxel6_subWarp <<< G3, B5 >>>(*positionFieldImageArray_d, *controlPointImageArray_d);
    else if (B5.x <= 1024)
        reg_bspline_getDeformationFieldTileVoxel6 <<< G3, B5 >>>(*positionFieldImageArray_d, *controlPointImageArray_d);
    else if (B6.x <= 896) // Can't run more threads because of limited registers (67 per thread for this arch)
        reg_bspline_getDeformationFieldTileVoxel7 <<< G3, B6 >>>(*positionFieldImageArray_d, *controlPointImageArray_d);
    else executed = false;
#endif
#if ID == 5
    reg_bspline_getDeformationField0_noMask <<< G0, B0 >>>(*positionFieldImageArray_d);
#endif

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_box = 0;
    cudaEventElapsedTime(&ms_box, start, stop);

    printf("**** Bspline kernel time: %fms\n", ms_box);

    NR_CUDA_CHECK_KERNEL(G1,B1)

    const size_t floatCountGPU(activeVoxelNumber);
    const size_t memoryReqGPU(floatCountGPU * sizeof(float4));

    float4* hostDataFloat4 = new float4[floatCountGPU];

    cudaMemcpy(hostDataFloat4, *positionFieldImageArray_d, memoryReqGPU, cudaMemcpyDeviceToHost);

    float4* mem_test = new float4[floatCountGPU];
    float4 *mem_test_d;
    NR_CUDA_SAFE_CALL(cudaMallocHost((void**)&mem_test_d, voxelNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(mem_test_d, *positionFieldImageArray_d, memoryReqGPU, cudaMemcpyDeviceToDevice))

    cudaEventRecord(start);
    cudaMemcpy(mem_test, mem_test_d, memoryReqGPU, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_memout = 0;
    cudaEventElapsedTime(&ms_memout, start, stop);

    if (!executed) ms_box = 9999.f;
    fprintf(f, "%d,%d,%f,%f,%f,", ID, level, ms_memin, ms_box, ms_memout);

#endif

    /************************ test correctness *****************************/
    reg_bspline_getDeformationField0 <<< G0, B0 >>>(*positionFieldImageArray_d);

    const size_t floatCount(activeVoxelNumber);
    const size_t memoryReq(floatCount * sizeof(float4));
    float4* hostDataFloat4Ground = new float4[floatCount];
    NR_CUDA_SAFE_CALL(cudaMemcpy(hostDataFloat4Ground, *positionFieldImageArray_d, memoryReq, cudaMemcpyDeviceToHost))

    /************************ test accuracy *****************************/
    double4_t * positionFieldImageArrayDouble_h = (double4_t *)calloc(
            referenceImageDimCPU.x*referenceImageDimCPU.y*referenceImageDimCPU.z , sizeof(double4_t));
    reg_bspline_getDeformationFieldLerpCPU2_doubleOUT(controlPointImageArray_h, positionFieldImageArrayDouble_h,
            controlPointImageDimCPU, referenceImageDimCPU, controlPointVoxelSpacingCPU);

    /************************ run the tests *****************************/

    int errors = 0;
    double3_t average_error = make_double3_t(0.0, 0.0, 0.0);
    int count_x = 0,count_y = 0,count_z = 0;
    float e = 1.f;
    for (int i = 0; i < activeVoxelNumber; ++i) {
        average_error.x = average_error.x + fabs( (double)hostDataFloat4[i].x - positionFieldImageArrayDouble_h[i].x );
        average_error.y = average_error.y + fabs( (double)hostDataFloat4[i].y - positionFieldImageArrayDouble_h[i].y );
        average_error.z = average_error.z + fabs( (double)hostDataFloat4[i].z - positionFieldImageArrayDouble_h[i].z );

        errors += fabs( hostDataFloat4[i].x - hostDataFloat4Ground[i].x ) > e;
        errors += fabs( hostDataFloat4[i].y - hostDataFloat4Ground[i].y ) > e;
        errors += fabs( hostDataFloat4[i].z - hostDataFloat4Ground[i].z ) > e;
        errors += fabs( hostDataFloat4[i].w - hostDataFloat4Ground[i].w ) > e;
        if (fabs( hostDataFloat4[i].x - hostDataFloat4Ground[i].x ) > e){
            count_x++;
        }
        if (fabs( hostDataFloat4[i].y - hostDataFloat4Ground[i].y ) > e){
            count_y++;
        }
        if (fabs( hostDataFloat4[i].z - hostDataFloat4Ground[i].z ) > e){
            count_z++;
        }
    }
    average_error = average_error / activeVoxelNumber;
    double average_xyz = (average_error.x + average_error.y + average_error.z) / 3;

    fprintf(f, "%.10f,%d,%d,%d,%d\n", average_xyz, errors, controlPointVoxelSpacingInt.x, controlPointVoxelSpacingInt.y, controlPointVoxelSpacingInt.z);

    printf("**** Average error in x: %f, y: %f, z: %f\n", average_error.x, average_error.y, average_error.z);
    printf("**** Number of errors in x: %d, y: %d, z: %d\n", count_x, count_y, count_z);
    printf("**** Number of misalignment errors: %d\n", errors);

    fclose(f);

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))

    NR_CUDA_SAFE_CALL(cudaFreeArray(controlPoints3D))
    delete[] hostDataFloat4;
    delete[] hostDataFloat4Ground;
    delete[] controlTemp;
    free(controlPointImageArray_h);
    free(positionFieldImageArray_h);
    free(positionFieldImageArrayDouble_h);
#if ID > 0
    delete[] mem_test;
    NR_CUDA_SAFE_CALL(cudaFreeArray(input3D))
    NR_CUDA_SAFE_CALL(cudaFree(preheat))
    NR_CUDA_SAFE_CALL(cudaFreeHost(mem_test_d))
#endif

    return;
}
/* *************************************************************** */
/* **************** Here we compare the performance of our approach in registration ***************** */
void reg_bspline_test_reg(nifti_image *controlPointImage,
                     nifti_image *reference,
                     float4 **controlPointImageArray_d,
                     float4 **positionFieldImageArray_d,
                     int **mask_d,
                     int activeVoxelNumber,
                     bool bspline)
{

#define IDR 2 //findtoo, valid values = 1,2: 1 original, 2 accelerated - check also reg_apps/reg_f3d.cpp if necessary

#if IDR == 1
    const int useBSpline = bspline;
#endif

    const int voxelNumber = reference->nx * reference->ny * reference->nz;
    const int3 referenceImageDim = make_int3(reference->nx, reference->ny, reference->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);

    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / reference->dx,
        controlPointImage->dy / reference->dy,
        controlPointImage->dz / reference->dz);

#if IDR == 2
    const int3 controlPointVoxelSpacingInt = make_int3(
        lrintf( controlPointImage->dx / reference->dx ),
        lrintf( controlPointImage->dy / reference->dy ),
        lrintf( controlPointImage->dz / reference->dz ));
#endif

#if IDR == 1
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_UseBSpline,&useBSpline,sizeof(int)))//
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))//
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))//
#endif
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))//
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))//
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))

#if IDR == 2
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_controlPointVoxelSpacingInt,&controlPointVoxelSpacingInt,sizeof(float3)))
#endif

    NR_CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))

#if IDR == 2
    const int3 tilesDim = make_int3(ceilf((float)reference->nx / (float)controlPointVoxelSpacing.x),
                                    ceilf((float)reference->ny / (float)controlPointVoxelSpacing.y),
                                    ceilf((float)reference->nz / (float)controlPointVoxelSpacing.z));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_tilesDim,&tilesDim,sizeof(int3)))

    float x_g0[MAX_CURRENT_SPACE];
    float y_g0[MAX_CURRENT_SPACE];
    float z_g0[MAX_CURRENT_SPACE];
    float x_h0_r[MAX_CURRENT_SPACE];
    float x_h1_r[MAX_CURRENT_SPACE];
    float y_h0_r[MAX_CURRENT_SPACE];
    float y_h1_r[MAX_CURRENT_SPACE];
    float z_h0_r[MAX_CURRENT_SPACE];
    float z_h1_r[MAX_CURRENT_SPACE];
    float xBasis[NUM_C*MAX_CURRENT_SPACE];
    float yBasis[NUM_C*MAX_CURRENT_SPACE];
    float zBasis[NUM_C*MAX_CURRENT_SPACE];
    float relative;
    for (int i = 0; i < controlPointVoxelSpacingInt.x; ++i) {
        relative = (float) i / controlPointVoxelSpacing.x;
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.f-relative;
        xBasis[0+NUM_C*i] = (MF)*(MF)*(MF)/(6.f);
        xBasis[1+NUM_C*i] = (3.f*FFF - 6.f*FF + 4.f)/6.f;
        xBasis[2+NUM_C*i] = (-3.f*FFF + 3.f*FF + 3.f*relative + 1.f)/6.f;
        xBasis[3+NUM_C*i] = (FFF/6.f);

        x_g0[i] = xBasis[0+NUM_C*i] + xBasis[1+NUM_C*i];

        x_h0_r[i] = xBasis[1+NUM_C*i] / (x_g0[i]);
        x_h1_r[i] = xBasis[3+NUM_C*i] / (1 - x_g0[i]);

    }
    for (int i = 0; i < controlPointVoxelSpacingInt.y; ++i) {
        relative = (float) i / controlPointVoxelSpacing.y;
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.f-relative;
        yBasis[0+NUM_C*i] = (MF)*(MF)*(MF)/(6.f);
        yBasis[1+NUM_C*i] = (3.f*FFF - 6.f*FF + 4.f)/6.f;
        yBasis[2+NUM_C*i] = (-3.f*FFF + 3.f*FF + 3.f*relative + 1.f)/6.f;
        yBasis[3+NUM_C*i] = (FFF/6.f);

        y_g0[i] = yBasis[0+NUM_C*i] + yBasis[1+NUM_C*i];

        y_h0_r[i] = yBasis[1+NUM_C*i] / (y_g0[i]);
        y_h1_r[i] = yBasis[3+NUM_C*i] / (1 - y_g0[i]);
    }
    for (int i = 0; i < controlPointVoxelSpacingInt.z; ++i) {
        relative = (float) i / controlPointVoxelSpacing.z;
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.f-relative;
        zBasis[0+NUM_C*i] = (MF)*(MF)*(MF)/(6.f);
        zBasis[1+NUM_C*i] = (3.f*FFF - 6.f*FF + 4.f)/6.f;
        zBasis[2+NUM_C*i] = (-3.f*FFF + 3.f*FF + 3.f*relative + 1.f)/6.f;
        zBasis[3+NUM_C*i] = (FFF/6.f);

        z_g0[i] = zBasis[0+NUM_C*i] + zBasis[1+NUM_C*i];

        z_h0_r[i] = zBasis[1+NUM_C*i] / (z_g0[i]);
        z_h1_r[i] = zBasis[3+NUM_C*i] / (1 - z_g0[i]);
    }

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_g0,&x_g0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_g0,&y_g0,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_g0,&z_g0,MAX_CURRENT_SPACE*sizeof(float)))

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_h0_r,&x_h0_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x_h1_r,&x_h1_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_h0_r,&y_h0_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_y_h1_r,&y_h1_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_h0_r,&z_h0_r,MAX_CURRENT_SPACE*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_z_h1_r,&z_h1_r,MAX_CURRENT_SPACE*sizeof(float)))

    dim3 B2(BLOCK_BS_X,BLOCK_BS_Y,BLOCK_BS_Z);
    dim3 G2((unsigned int) ceilf((float)(tilesDim.x) / B2.x),
            (unsigned int) ceilf((float)(tilesDim.y) / B2.y),
            (unsigned int) ceilf((float)(tilesDim.z) / B2.z));
#endif

#if IDR == 1
    const unsigned int Grid_reg_bspline_getDeformationField0 =
        (unsigned int)ceilf(sqrtf((float)activeVoxelNumber/(float)(Block_reg_bspline_getDeformationField)));
    dim3 G0(Grid_reg_bspline_getDeformationField0,Grid_reg_bspline_getDeformationField0,1);
    dim3 B0(256,1,1);
#endif

#if IDR == 1
    reg_bspline_getDeformationField0_noMask<<< G0, B0 >>>(*positionFieldImageArray_d);
#endif
#if IDR == 2
    reg_bspline_getDeformationFieldTileLerp3_noSh<<<G2, B2 >>>(*positionFieldImageArray_d, *controlPointImageArray_d);
#endif

    cudaThreadSynchronize();
    cudaError err = cudaPeekAtLastError();
    if( err != cudaSuccess) {
        fprintf(stderr, "[NiftyReg CUDA ERROR] file _reg_localTransformation_gpu.cu : %s.\n",
        cudaGetErrorString(err));
        exit(1);
    }

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    return;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_bspline_ApproxBendingEnergy_gpu(nifti_image *controlPointImage,
                                          float4 **controlPointImageArray_d)
{
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointGridMem))
    const unsigned int Grid_bspline_getApproxSecondDerivatives =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxSecondDerivatives)));
    dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
    dim3 B1(Block_reg_bspline_getApproxSecondDerivatives,1,1);
    reg_bspline_getApproxSecondDerivatives <<< G1, B1 >>>(secondDerivativeValues_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the bending energy from the second derivatives
    float *penaltyTerm_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&penaltyTerm_d, controlPointNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                      secondDerivativeValues_d,
                                      6*controlPointGridMem))
    const unsigned int Grid_reg_bspline_ApproxBendingEnergy =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxBendingEnergy)));
    dim3 G2(Grid_reg_bspline_ApproxBendingEnergy,Grid_reg_bspline_ApproxBendingEnergy,1);
    dim3 B2(Block_reg_bspline_getApproxBendingEnergy,1,1);
    reg_bspline_getApproxBendingEnergy_kernel <<< G2, B2 >>>(penaltyTerm_d);
    NR_CUDA_CHECK_KERNEL(G2,B2)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValues_d))

    // Transfert the vales back to the CPU and average them
    float *penaltyTerm_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&penaltyTerm_h, controlPointNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(penaltyTerm_h, penaltyTerm_d, controlPointNumber*sizeof(float), cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(penaltyTerm_d))

    double penaltyValue=0.0;
    for(int i=0;i<controlPointNumber;i++)
            penaltyValue += penaltyTerm_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)penaltyTerm_h))
    return (float)(penaltyValue/(3.0*(double)controlPointNumber));
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ApproxBendingEnergyGradient_gpu(nifti_image *referenceImage,
                                                 nifti_image *controlPointImage,
                                                 float4 **controlPointImageArray_d,
                                                 float4 **nodeNMIGradientArray_d,
                                                 float bendingEnergyWeight)
{
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointNumber*sizeof(float4)))
    const unsigned int Grid_bspline_getApproxSecondDerivatives =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxSecondDerivatives)));
    dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
    dim3 B1(Block_reg_bspline_getApproxSecondDerivatives,1,1);
    reg_bspline_getApproxSecondDerivatives <<< G1, B1 >>>(secondDerivativeValues_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the gradient
    bendingEnergyWeight *= referenceImage->nx*referenceImage->ny*referenceImage->nz /
                           (controlPointImage->nx*controlPointImage->ny*controlPointImage->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&bendingEnergyWeight,sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                      secondDerivativeValues_d,
                                      6*controlPointNumber*sizeof(float4)))
    const unsigned int Grid_reg_bspline_getApproxBendingEnergyGradient =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxBendingEnergyGradient)));
    dim3 G2(Grid_reg_bspline_getApproxBendingEnergyGradient,Grid_reg_bspline_getApproxBendingEnergyGradient,1);
    dim3 B2(Block_reg_bspline_getApproxBendingEnergyGradient,1,1);
    reg_bspline_getApproxBendingEnergyGradient_kernel <<< G2, B2 >>>(*nodeNMIGradientArray_d);
    NR_CUDA_CHECK_KERNEL(G2,B2)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValues_d))

    return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ComputeApproxJacobianValues(nifti_image *controlPointImage,
                                             float4 **controlPointImageArray_d,
                                             float **jacobianMatrices_d,
                                             float **jacobianDet_d)
{
    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // The Jacobian matrix is computed for every control point
    const unsigned int Grid_reg_bspline_getApproxJacobianValues =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxJacobianValues)));
    dim3 G1(Grid_reg_bspline_getApproxJacobianValues,Grid_reg_bspline_getApproxJacobianValues,1);
    dim3 B1(Block_reg_bspline_getApproxJacobianValues,1,1);
    reg_bspline_getApproxJacobianValues_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
void reg_bspline_ComputeJacobianValues(nifti_image *controlPointImage,
                                       nifti_image *referenceImage,
                                       float4 **controlPointImageArray_d,
                                       float **jacobianMatrices_d,
                                       float **jacobianDet_d)
{
    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    // Bind some variables
    const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    const float3 controlPointVoxelSpacing = make_float3(
            controlPointImage->dx / referenceImage->dx,
            controlPointImage->dy / referenceImage->dy,
            controlPointImage->dz / referenceImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))

    // The Jacobian matrix is computed for every voxel
    const unsigned int Grid_reg_bspline_getJacobianValues =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(Block_reg_bspline_getJacobianValues)));
    dim3 G1(Grid_reg_bspline_getJacobianValues,Grid_reg_bspline_getJacobianValues,1);
    dim3 B1(Block_reg_bspline_getJacobianValues,1,1);
    reg_bspline_getJacobianValues_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
/* *************************************************************** */
double reg_bspline_ComputeJacobianPenaltyTerm_gpu(nifti_image *referenceImage,
                                                  nifti_image *controlPointImage,
                                                  float4 **controlPointImageArray_d,
                                                  bool approx
                                                  )
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    double jacSum;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        jacSum=(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        jacSum=jacNumber;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))

    // The Jacobian determinant are squared and logged (might not be english but will do)
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&jacNumber,sizeof(int)))
    const unsigned int Grid_reg_bspline_logSquaredValues =
        (unsigned int)ceilf(sqrtf((float)jacNumber/(float)(Block_reg_bspline_logSquaredValues)));
    dim3 G1(Grid_reg_bspline_logSquaredValues,Grid_reg_bspline_logSquaredValues,1);
    dim3 B1(Block_reg_bspline_logSquaredValues,1,1);
    reg_bspline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    // Transfert the data back to the CPU
    float *jacobianDet_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&jacobianDet_h,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet_h,jacobianDet_d,
                                 jacNumber*sizeof(float),
                                 cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    double penaltyTermValue=0.;
    for(int i=0;i<jacNumber;++i)
        penaltyTermValue += jacobianDet_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost(jacobianDet_h))
    return penaltyTermValue/jacSum;
}
/* *************************************************************** */
void reg_bspline_ComputeJacobianPenaltyTermGradient_gpu(nifti_image *referenceImage,
                                                        nifti_image *controlPointImage,
                                                        float4 **controlPointImageArray_d,
                                                        float4 **nodeNMIGradientArray_d,
                                                        float jacobianWeight,
                                                        bool approx)
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }

    // Need to desorient the Jacobian matrix using the header information - voxel to real conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianDeterminantTexture, jacobianDet_d,
                                      jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianMatricesTexture, jacobianMatrices_d,
                                      9*jacNumber*sizeof(float)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    if(approx){
        float weight=jacobianWeight;
        weight = jacobianWeight * (float)(referenceImage->nx * referenceImage->ny * referenceImage->nz)
                 / (float)( controlPointImage->nx*controlPointImage->ny*controlPointImage->nz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&weight,sizeof(float)))
        const unsigned int Grid_reg_bspline_computeApproxJacGradient =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_computeApproxJacGradient)));
        dim3 G1(Grid_reg_bspline_computeApproxJacGradient,Grid_reg_bspline_computeApproxJacGradient,1);
        dim3 B1(Block_reg_bspline_computeApproxJacGradient,1,1);
        reg_bspline_computeApproxJacGradient_kernel<<< G1, B1>>>(*nodeNMIGradientArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(
                controlPointImage->dx / referenceImage->dx,
                controlPointImage->dy / referenceImage->dy,
                controlPointImage->dz / referenceImage->dz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&jacobianWeight,sizeof(float)))
        const unsigned int Grid_reg_bspline_computeJacGradient =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_computeJacGradient)));
        dim3 G1(Grid_reg_bspline_computeJacGradient,Grid_reg_bspline_computeJacGradient,1);
        dim3 B1(Block_reg_bspline_computeJacGradient,1,1);
        reg_bspline_computeJacGradient_kernel<<< G1, B1>>>(*nodeNMIGradientArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianDeterminantTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianMatricesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
}
/* *************************************************************** */
double reg_bspline_correctFolding_gpu(nifti_image *referenceImage,
                                      nifti_image *controlPointImage,
                                      float4 **controlPointImageArray_d,
                                      bool approx)
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    double jacSum;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        jacSum = (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacSum=jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }

    // Check if the Jacobian determinant average
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&jacNumber,sizeof(int)))
    float *jacobianDet2_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet2_d,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet2_d,jacobianDet_d,jacNumber*sizeof(float),cudaMemcpyDeviceToDevice))
    const unsigned int Grid_reg_bspline_logSquaredValues =
        (unsigned int)ceilf(sqrtf((float)jacNumber/(float)(Block_reg_bspline_logSquaredValues)));
    dim3 G1(Grid_reg_bspline_logSquaredValues,Grid_reg_bspline_logSquaredValues,1);
    dim3 B1(Block_reg_bspline_logSquaredValues,1,1);
    reg_bspline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet2_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    float *jacobianDet_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&jacobianDet_h,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet_h,jacobianDet2_d,
                                 jacNumber*sizeof(float),
                                 cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet2_d))
    double penaltyTermValue=0.;
    for(int i=0;i<jacNumber;++i) penaltyTermValue += jacobianDet_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost(jacobianDet_h))
    penaltyTermValue /= jacSum;
    if(penaltyTermValue==penaltyTermValue){
        NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
        NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
        return penaltyTermValue;
    }

    // Need to desorient the Jacobian matrix using the header information - voxel to real conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianDeterminantTexture, jacobianDet_d,
                                      jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianMatricesTexture, jacobianMatrices_d,
                                      9*jacNumber*sizeof(float)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    if(approx){
        const unsigned int Grid_reg_bspline_approxCorrectFolding =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_approxCorrectFolding)));
        dim3 G1(Grid_reg_bspline_approxCorrectFolding,Grid_reg_bspline_approxCorrectFolding,1);
        dim3 B1(Block_reg_bspline_approxCorrectFolding,1,1);
        reg_bspline_approxCorrectFolding_kernel<<< G1, B1>>>(*controlPointImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(
                controlPointImage->dx / referenceImage->dx,
                controlPointImage->dy / referenceImage->dy,
                controlPointImage->dz / referenceImage->dz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
        const unsigned int Grid_reg_bspline_correctFolding =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_correctFolding)));
        dim3 G1(Grid_reg_bspline_correctFolding,Grid_reg_bspline_correctFolding,1);
        dim3 B1(Block_reg_bspline_correctFolding,1,1);
        reg_bspline_correctFolding_kernel<<< G1, B1>>>(*controlPointImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianDeterminantTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianMatricesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDeformationFromDisplacement_gpu( nifti_image *image, float4 **imageArray_d)
{
    // Bind the qform or sform
    mat44 temp_mat=image->qto_xyz;
    if(image->sform_code>0) temp_mat=image->sto_xyz;
    float4 temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2b,&temp,sizeof(float4)))

    const int voxelNumber=image->nx*image->ny*image->nz;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))

    const int3 imageDim=make_int3(image->nx,image->ny,image->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&imageDim,sizeof(int3)))

    const unsigned int Grid_reg_getDeformationFromDisplacement =
    (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(512)));
    dim3 G1(Grid_reg_getDeformationFromDisplacement,Grid_reg_getDeformationFromDisplacement,1);
    dim3 B1(512,1,1);
    reg_getDeformationFromDisplacement_kernel<<< G1, B1>>>(*imageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDisplacementFromDeformation_gpu( nifti_image *image, float4 **imageArray_d)
{
    // Bind the qform or sform
    mat44 temp_mat=image->qto_xyz;
    if(image->sform_code>0) temp_mat=image->sto_xyz;
    float4 temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2b,&temp,sizeof(float4)))

    const int voxelNumber=image->nx*image->ny*image->nz;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))

    const int3 imageDim=make_int3(image->nx,image->ny,image->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&imageDim,sizeof(int3)))

    const unsigned int Grid_reg_getDisplacementFromDeformation =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(512)));
    dim3 G1(Grid_reg_getDisplacementFromDeformation,Grid_reg_getDisplacementFromDeformation,1);
    dim3 B1(512,1,1);
    reg_getDisplacementFromDeformation_kernel<<< G1, B1>>>(*imageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDeformationFieldFromVelocityGrid_gpu(nifti_image *cpp_h,
                                                 nifti_image *def_h,
                                                 float4 **cpp_gpu,
                                                 float4 **def_gpu,
                                                 float4 **interDef_gpu,
                                                 int **mask_gpu,
                                                 int activeVoxel,
                                                 bool approxComp)
{
    if(approxComp){
        fprintf(stderr, "[NiftyReg] reg_getDeformationFieldFromVelocityGrid_gpu\n");
        fprintf(stderr, "[NiftyReg] ERROR Approximation not implemented yet on the GPU\n");
        exit(1);
    }

    const int controlPointNumber = cpp_h->nx * cpp_h->ny * cpp_h->nz;
    const int voxelNumber = def_h->nx * def_h->ny * def_h->nz;

    if(voxelNumber != activeVoxel){
        fprintf(stderr, "[NiftyReg] reg_getDeformationFieldFromVelocityGrid_gpu\n");
        fprintf(stderr, "[NiftyReg] ERROR The mask must contains all voxel\n");
        exit(1);
    }

    // A scaled down velocity field is first store
    float4 *scaledVelocityField_d=NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&scaledVelocityField_d,controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(scaledVelocityField_d,*cpp_gpu,controlPointNumber*sizeof(float4),cudaMemcpyDeviceToDevice))
    reg_getDisplacementFromDeformation_gpu(cpp_h, &scaledVelocityField_d);
    reg_multiplyValue_gpu(controlPointNumber,&scaledVelocityField_d,1.f/cpp_h->pixdim[5]);
    reg_getDeformationFromDisplacement_gpu(cpp_h, &scaledVelocityField_d);

    if(!approxComp){
        float4 *tempDef=NULL;
        float4 *currentDefPtr0=NULL;
        float4 *currentDefPtr1=NULL;
        if(interDef_gpu==NULL){
            NR_CUDA_SAFE_CALL(cudaMalloc(&tempDef,voxelNumber*sizeof(float4)))
            currentDefPtr0 = *def_gpu;
            currentDefPtr1 = tempDef;
        }
        else{
            currentDefPtr0 = interDef_gpu[0];
            currentDefPtr1 = interDef_gpu[1];
        }
        reg_bspline_gpu(cpp_h,
                        def_h,
                        &scaledVelocityField_d,
                        &currentDefPtr0,
                        mask_gpu,
                        activeVoxel,
                        true);

        for(unsigned int i=0;i<cpp_h->pixdim[5];++i){

            NR_CUDA_SAFE_CALL(cudaMemcpy(currentDefPtr1,currentDefPtr0,voxelNumber*sizeof(float4),cudaMemcpyDeviceToDevice))

            if(interDef_gpu==NULL){
                reg_defField_compose_gpu(def_h,
                                         &currentDefPtr1,
                                         &currentDefPtr0,
                                         mask_gpu,
                                         activeVoxel);
            }
            else{
                reg_defField_compose_gpu(def_h,
                                         &currentDefPtr0,
                                         &currentDefPtr1,
                                         mask_gpu,
                                         activeVoxel);
                if(i==cpp_h->pixdim[5]-2){
                    currentDefPtr0 = interDef_gpu[i+1];
                    currentDefPtr1 = *def_gpu;
                }
                else if(i<cpp_h->pixdim[5]-2){
                    currentDefPtr0 = interDef_gpu[i+1];
                    currentDefPtr1 = interDef_gpu[i+2];
                }
            }
        }
        if(tempDef!=NULL) NR_CUDA_SAFE_CALL(cudaFree(tempDef));
    }
    NR_CUDA_SAFE_CALL(cudaFree(scaledVelocityField_d))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getInverseDeformationFieldFromVelocityGrid_gpu(nifti_image *cpp_h,
                                                        nifti_image *def_h,
                                                        float4 **cpp_gpu,
                                                        float4 **def_gpu,
                                                        float4 **interDef_gpu,
                                                        int **mask_gpu,
                                                        int activeVoxel,
                                                        bool approxComp)
{
    const int controlPointNumber = cpp_h->nx * cpp_h->ny * cpp_h->nz;
    // The CPP file is first negated
    float4 *invertedCpp_gpu=NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&invertedCpp_gpu,controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(invertedCpp_gpu,*cpp_gpu,controlPointNumber*sizeof(float4),cudaMemcpyDeviceToDevice))
    reg_getDisplacementFromDeformation_gpu(cpp_h, &invertedCpp_gpu);
    reg_multiplyValue_gpu(controlPointNumber,&invertedCpp_gpu,-1.f);
    reg_getDeformationFromDisplacement_gpu(cpp_h, &invertedCpp_gpu);

    reg_getDeformationFieldFromVelocityGrid_gpu(cpp_h,
                                                def_h,
                                                &invertedCpp_gpu,
                                                def_gpu,
                                                interDef_gpu,
                                                mask_gpu,
                                                activeVoxel,
                                                approxComp);
    NR_CUDA_SAFE_CALL(cudaFree(invertedCpp_gpu))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_compose_gpu(nifti_image *def,
                              float4 **def_gpu,
                              float4 **defOut_gpu,
                              int **mask_gpu,
                              int activeVoxel)
{
    const int voxelNumber=def->nx*def->ny*def->nz;
    if(voxelNumber != activeVoxel){
        fprintf(stderr, "[NiftyReg] reg_defField_compose_gpu\n");
        fprintf(stderr, "[NiftyReg] ERROR no mask can be used\n");
        exit(1);
    }

    // Bind the qform or sform
    mat44 temp_mat=def->qto_ijk;
    if(def->sform_code>0) temp_mat=def->sto_ijk;
    float4 temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2b,&temp,sizeof(float4)))

    temp_mat=def->qto_xyz;
    if(def->sform_code>0) temp_mat=def->sto_xyz;
    temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0c,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1c,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2c,&temp,sizeof(float4)))

    const int3 referenceImageDim=make_int3(def->nx,def->ny,def->nz);

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,voxelDisplacementTexture,*def_gpu,activeVoxel*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,maskTexture,*mask_gpu,activeVoxel*sizeof(int)))

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int)))

    const unsigned int Grid_reg_defField_compose =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(Block_reg_defField_compose)));
    dim3 G1(Grid_reg_defField_compose,Grid_reg_defField_compose,1);
    dim3 B1(Block_reg_defField_compose,1,1);
    reg_defField_compose_kernel<<< G1, B1>>>(*defOut_gpu);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(voxelDisplacementTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_getJacobianMatrix_gpu(nifti_image *deformationField,
                                        float4 **deformationField_gpu,
                                        float **jacobianMatrices_gpu)
{
    const int3 referenceDim=make_int3(deformationField->nx,deformationField->ny,deformationField->nz);
    const float3 referenceSpacing=make_float3(deformationField->dx,deformationField->dy,deformationField->dz);
    const int voxelNumber = referenceDim.x*referenceDim.y*referenceDim.z;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceSpacing,&referenceSpacing,sizeof(float3)))

    mat33 reorient, desorient;
    reg_getReorientationMatrix(deformationField, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,voxelDisplacementTexture,*deformationField_gpu,voxelNumber*sizeof(float4)))

    const unsigned int Grid_reg_defField_getJacobianMatrix =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(Block_reg_defField_getJacobianMatrix)));
    dim3 G1(Grid_reg_defField_getJacobianMatrix,Grid_reg_defField_getJacobianMatrix,1);
    dim3 B1(Block_reg_defField_getJacobianMatrix);
    reg_defField_getJacobianMatrix_kernel<<<G1,B1>>>(*jacobianMatrices_gpu);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(voxelDisplacementTexture))
}
/* *************************************************************** */
/* *************************************************************** */
#endif
