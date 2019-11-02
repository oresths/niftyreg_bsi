# Accelerating B-spline Interpolation On Gpus: Application To Medical Image Registration
In this repository we provide the source code of our accelerated B-spline interpolation implementations, which are described in our paper "Accelerating B-spline Interpolation On Gpus: Application To Medical Image Registration".

## Structure
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
<Placeholder for citing the paper and the research data>

## For further information please have a look at the original README.txt
