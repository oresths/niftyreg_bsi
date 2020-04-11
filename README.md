# Accelerating B-spline Interpolation On Gpus: Application To Medical Image Registration
In this repository we provide the source code of our accelerated B-spline interpolation implementations, which are described in our paper "Accelerating B-spline Interpolation On Gpus: Application To Medical Image Registration" [1]. The B-spline interpolation implementations are integrated into Nifty Reg medical image registration library [3]. We also provide our image registration dataset [2].

## Structure
Our B-spline implementations for GPU are mainly in 2 files:
reg-lib/_reg_localTransformation_gpu.cu
reg-lib/_reg_localTransformation_kernels.cu 

Our B-spline implementations for CPU are mainly in:
reg-lib/_reg_localTransformation.cpp

All our modifications to the original Nifty Reg source code have been aggregated to a single commit (2nd commit in the commit history). Thus, the common git tools can be used to spot all our modifications.

We provide our code in two branches, master and tests.

master: In this branch, we replace the original B-spline interpolation implementations with our "Thread per Tile with Linear Interpolations" and "Vector per Tile" implementations, for GPU and CPU respectively. Compile this branch if you just want to perform image registration.

tests: This branch has the code that we used to evaluate our approach. tests works only on Linux.

## How to build
Cmake and CUDA are required. Call cmake like this:

"cmake -D CMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_OPENMP=ON -DUSE_SSE=ON ../niftyreg_bsi"

Note that it may be necessary to set the CUDA_TOOLKIT_ROOT_DIR and CUDA_SDK_ROOT_DIR cmake flags.

## How to run
In order to test image registration run reg_f3d app inside reg-apps folder.

### example
./reg-apps/reg_f3d -ref dataset/0.nii.gz -flo dataset/1.nii.gz -sx -5

## Current limitations
1. Voxel mask is not supported. Please avoid the use of "-rmask" option.
2. Non-integer grid spacing is not functional. Please use "-sx", "-sy", "-sz" only with negative numbers.
3. Vector per Tile CPU implementation supports a maximum of 16 voxels in grid spacing along the x axis.

## Contact data
Orestis Zachariadis (orestis.zachariadis@uco.es)

## Acknowledgements
Marc Modat for creating an excellent medical image registration library.

## References
[1]O. Zachariadis et al., “Accelerating B-spline interpolation on GPUs: Application to medical image registration,” Computer Methods and Programs in Biomedicine, vol. 193, p. 105431, Sep. 2020, doi: 10.1016/j.cmpb.2020.105431.

[2]Jakob Elle, Ole; Teatini, Andrea; Zachariadis, Orestis (2020), “Data for: Accelerating B-spline Interpolation on GPUs: Application to Medical Image Registration”, Mendeley Data, v1 http://dx.doi.org/10.17632/kj3xcd776k.1

[3]M. Modat et al., “Fast free-form deformation using graphics processing units,” Computer Methods and Programs in Biomedicine, vol. 98, no. 3, pp. 278–284, 2010, doi: 10.1016/j.cmpb.2009.09.002.


## For further information please have a look at the original README.txt
