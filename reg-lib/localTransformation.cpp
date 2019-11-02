#include "localTransformation.h"

// Daniels's calculation of the bspline convolution weights
void bspline_weights_cpu(float3_t fraction, float3_t& w0, float3_t& w1, float3_t& w2, float3_t& w3)
{
    const float3_t one_frac = 1.0f - fraction;
    const float3_t squared = fraction * fraction;
    const float3_t one_sqd = one_frac * one_frac;

    w0 = 1.0f/6.0f * one_sqd * one_frac;
    w1 = 2.0f/3.0f - 0.5f * squared * (2.0f-fraction);
    w2 = 2.0f/3.0f - 0.5f * one_sqd * (2.0f-one_frac);
    w3 = 1.0f/6.0f * squared * fraction;
}
void bspline_weights_cpu_double(double3_t fraction, double3_t& w0, double3_t& w1, double3_t& w2, double3_t& w3)
{
    const double3_t one_frac = 1.0 - fraction;
    const double3_t squared = fraction * fraction;
    const double3_t one_sqd = one_frac * one_frac;

    w0 = 1.0/6.0 * one_sqd * one_frac;
    w1 = 2.0/3.0 - 0.5 * squared * (2.0-fraction);
    w2 = 2.0/3.0 - 0.5 * one_sqd * (2.0-one_frac);
    w3 = 1.0/6.0 * squared * fraction;
}
/* ************************************************************** */

/* ******* Double precision implementation (Fused Multiply Accumulate) that is used for accuracy comparison ******** */
void reg_bspline_getDeformationFieldLerpCPU2_doubleOUT(float4_t *controlPoint, double4_t *positionField,
        const int3_t controlPointImageDim, const int3_t referenceImageDim, const float3_t controlPointVoxelSpacing)
{
    double3_t gridVoxelSpacing;
    gridVoxelSpacing.x = (double)controlPointVoxelSpacing.x;
    gridVoxelSpacing.y = (double)controlPointVoxelSpacing.y;
    gridVoxelSpacing.z = (double)controlPointVoxelSpacing.z;

    const int voxelNumber = controlPointImageDim.x * controlPointImageDim.y * controlPointImageDim.z;

    for (int z = 0; z < referenceImageDim.z; ++z) {
        for (int y = 0; y < referenceImageDim.y; ++y) {
            for (int x = 0; x < referenceImageDim.x; ++x) {

                // the "nearest previous" node is determined [0,0,0]
                double3_t nodeAnte;
                nodeAnte.x = floor((double)x/gridVoxelSpacing.x);
                nodeAnte.y = floor((double)y/gridVoxelSpacing.y);
                nodeAnte.z = floor((double)z/gridVoxelSpacing.z);

                double3_t relative;
                relative.z = fabs((double)z/gridVoxelSpacing.z-nodeAnte.z);
                relative.y = fabs((double)y/gridVoxelSpacing.y-nodeAnte.y);
                relative.x = fabs((double)x/gridVoxelSpacing.x-nodeAnte.x);

                double3_t w0, w1, w2, w3;
                bspline_weights_cpu_double(relative, w0, w1, w2, w3);

                const double3_t g0 = w0 + w1;
                const double3_t g1 = w2 + w3;
                const double3_t h0 = (w1 / g0);
                const double3_t h1 = (w3 / g1);

                double4_t nodeCoefficientA[NUM_C*NUM_C], nodeCoefficientB[NUM_C*NUM_C], nodeCoefficientC[NUM_C*NUM_C], nodeCoefficientD[NUM_C*NUM_C];

                for(int c=0; c<NUM_C; c++){
                    int indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
                    for(int b=0; b<NUM_C; b++){

                        int indexXYZ = indexYZ + nodeAnte.x;

                        nodeCoefficientA[b + NUM_C*c].x = (double)controlPoint[indexXYZ].x;
                        nodeCoefficientA[b + NUM_C*c].y = (double)controlPoint[indexXYZ].y;
                        nodeCoefficientA[b + NUM_C*c].z = (double)controlPoint[indexXYZ].z;
                        indexXYZ++;

                        nodeCoefficientB[b + NUM_C*c].x = (double)controlPoint[indexXYZ].x;
                        nodeCoefficientB[b + NUM_C*c].y = (double)controlPoint[indexXYZ].y;
                        nodeCoefficientB[b + NUM_C*c].z = (double)controlPoint[indexXYZ].z;
                        indexXYZ++;

                        nodeCoefficientC[b + NUM_C*c].x = (double)controlPoint[indexXYZ].x;
                        nodeCoefficientC[b + NUM_C*c].y = (double)controlPoint[indexXYZ].y;
                        nodeCoefficientC[b + NUM_C*c].z = (double)controlPoint[indexXYZ].z;
                        indexXYZ++;

                        nodeCoefficientD[b + NUM_C*c].x = (double)controlPoint[indexXYZ].x;
                        nodeCoefficientD[b + NUM_C*c].y = (double)controlPoint[indexXYZ].y;
                        nodeCoefficientD[b + NUM_C*c].z = (double)controlPoint[indexXYZ].z;

                        indexYZ += controlPointImageDim.x;
                    }
                }

                double4_t c000_,c001_,c010_,c011_,c100_,c101_,c110_,c111_;

                c000_ = nodeCoefficientA[0*NUM_C];
                c001_ = nodeCoefficientA[0*NUM_C+NUM_C];
                c010_ = nodeCoefficientA[0*NUM_C+1];
                c011_ = nodeCoefficientA[0*NUM_C+NUM_C+1];
                c100_ = nodeCoefficientB[0*NUM_C];
                c101_ = nodeCoefficientB[0*NUM_C+NUM_C];
                c110_ = nodeCoefficientB[0*NUM_C+1];
                c111_ = nodeCoefficientB[0*NUM_C+NUM_C+1];

                c000_ = c000_ + h0.z*(c001_-c000_);
                c010_ = c010_ + h0.z*(c011_-c010_);
                c100_ = c100_ + h0.z*(c101_-c100_);
                c110_ = c110_ + h0.z*(c111_-c110_);

                c000_ = c000_ + h0.y*(c010_-c000_);
                c100_ = c100_ + h0.y*(c110_-c100_);

                double4_t c000;
                c000 =  c000_ + h0.x*(c100_-c000_);

                c000_ = nodeCoefficientA[2*NUM_C];
                c001_ = nodeCoefficientA[2*NUM_C+NUM_C];
                c010_ = nodeCoefficientA[2*NUM_C+1];
                c011_ = nodeCoefficientA[2*NUM_C+NUM_C+1];
                c100_ = nodeCoefficientB[2*NUM_C];
                c101_ = nodeCoefficientB[2*NUM_C+NUM_C];
                c110_ = nodeCoefficientB[2*NUM_C+1];
                c111_ = nodeCoefficientB[2*NUM_C+NUM_C+1];

                c000_ = c000_ + h1.z*(c001_-c000_);
                c010_ = c010_ + h1.z*(c011_-c010_);
                c100_ = c100_ + h1.z*(c101_-c100_);
                c110_ = c110_ + h1.z*(c111_-c110_);

                c000_ = c000_ + h0.y*(c010_-c000_);
                c100_ = c100_ + h0.y*(c110_-c100_);

                double4_t c001;
                c001 =  c000_ + h0.x*(c100_-c000_);

                c000_ = nodeCoefficientA[0*NUM_C+2];
                c001_ = nodeCoefficientA[0*NUM_C+NUM_C+2];
                c010_ = nodeCoefficientA[0*NUM_C+1+2];
                c011_ = nodeCoefficientA[0*NUM_C+NUM_C+1+2];
                c100_ = nodeCoefficientB[0*NUM_C+2];
                c101_ = nodeCoefficientB[0*NUM_C+NUM_C+2];
                c110_ = nodeCoefficientB[0*NUM_C+1+2];
                c111_ = nodeCoefficientB[0*NUM_C+NUM_C+1+2];

                c000_ = c000_ + h0.z*(c001_-c000_);
                c010_ = c010_ + h0.z*(c011_-c010_);
                c100_ = c100_ + h0.z*(c101_-c100_);
                c110_ = c110_ + h0.z*(c111_-c110_);

                c000_ = c000_ + h1.y*(c010_-c000_);
                c100_ = c100_ + h1.y*(c110_-c100_);

                double4_t c010;
                c010 =  c000_ + h0.x*(c100_-c000_);

                c000_ = nodeCoefficientA[2*NUM_C+2];
                c001_ = nodeCoefficientA[2*NUM_C+NUM_C+2];
                c010_ = nodeCoefficientA[2*NUM_C+1+2];
                c011_ = nodeCoefficientA[2*NUM_C+NUM_C+1+2];
                c100_ = nodeCoefficientB[2*NUM_C+2];
                c101_ = nodeCoefficientB[2*NUM_C+NUM_C+2];
                c110_ = nodeCoefficientB[2*NUM_C+1+2];
                c111_ = nodeCoefficientB[2*NUM_C+NUM_C+1+2];

                c000_ = c000_ + h1.z*(c001_-c000_);
                c010_ = c010_ + h1.z*(c011_-c010_);
                c100_ = c100_ + h1.z*(c101_-c100_);
                c110_ = c110_ + h1.z*(c111_-c110_);

                c000_ = c000_ + h1.y*(c010_-c000_);
                c100_ = c100_ + h1.y*(c110_-c100_);

                double4_t c011;
                c011 =  c000_ + h0.x*(c100_-c000_);

                c000_ = nodeCoefficientC[0*NUM_C];
                c001_ = nodeCoefficientC[0*NUM_C+NUM_C];
                c010_ = nodeCoefficientC[0*NUM_C+1];
                c011_ = nodeCoefficientC[0*NUM_C+NUM_C+1];
                c100_ = nodeCoefficientD[0*NUM_C];
                c101_ = nodeCoefficientD[0*NUM_C+NUM_C];
                c110_ = nodeCoefficientD[0*NUM_C+1];
                c111_ = nodeCoefficientD[0*NUM_C+NUM_C+1];

                c000_ = c000_ + h0.z*(c001_-c000_);
                c010_ = c010_ + h0.z*(c011_-c010_);
                c100_ = c100_ + h0.z*(c101_-c100_);
                c110_ = c110_ + h0.z*(c111_-c110_);

                c000_ = c000_ + h0.y*(c010_-c000_);
                c100_ = c100_ + h0.y*(c110_-c100_);

                double4_t c100;
                c100 =  c000_ + h1.x*(c100_-c000_);

                c000_ = nodeCoefficientC[2*NUM_C];
                c001_ = nodeCoefficientC[2*NUM_C+NUM_C];
                c010_ = nodeCoefficientC[2*NUM_C+1];
                c011_ = nodeCoefficientC[2*NUM_C+NUM_C+1];
                c100_ = nodeCoefficientD[2*NUM_C];
                c101_ = nodeCoefficientD[2*NUM_C+NUM_C];
                c110_ = nodeCoefficientD[2*NUM_C+1];
                c111_ = nodeCoefficientD[2*NUM_C+NUM_C+1];

                c000_ = c000_ + h1.z*(c001_-c000_);
                c010_ = c010_ + h1.z*(c011_-c010_);
                c100_ = c100_ + h1.z*(c101_-c100_);
                c110_ = c110_ + h1.z*(c111_-c110_);

                c000_ = c000_ + h0.y*(c010_-c000_);
                c100_ = c100_ + h0.y*(c110_-c100_);

                double4_t c101;
                c101 =  c000_ + h1.x*(c100_-c000_);

                c000_ = nodeCoefficientC[0*NUM_C+2];
                c001_ = nodeCoefficientC[0*NUM_C+NUM_C+2];
                c010_ = nodeCoefficientC[0*NUM_C+1+2];
                c011_ = nodeCoefficientC[0*NUM_C+NUM_C+1+2];
                c100_ = nodeCoefficientD[0*NUM_C+2];
                c101_ = nodeCoefficientD[0*NUM_C+NUM_C+2];
                c110_ = nodeCoefficientD[0*NUM_C+1+2];
                c111_ = nodeCoefficientD[0*NUM_C+NUM_C+1+2];

                c000_ = c000_ + h0.z*(c001_-c000_);
                c010_ = c010_ + h0.z*(c011_-c010_);
                c100_ = c100_ + h0.z*(c101_-c100_);
                c110_ = c110_ + h0.z*(c111_-c110_);

                c000_ = c000_ + h1.y*(c010_-c000_);
                c100_ = c100_ + h1.y*(c110_-c100_);

                double4_t c110;
                c110 =  c000_ + h1.x*(c100_-c000_);

                c000_ = nodeCoefficientC[2*NUM_C+2];
                c001_ = nodeCoefficientC[2*NUM_C+NUM_C+2];
                c010_ = nodeCoefficientC[2*NUM_C+1+2];
                c011_ = nodeCoefficientC[2*NUM_C+NUM_C+1+2];
                c100_ = nodeCoefficientD[2*NUM_C+2];
                c101_ = nodeCoefficientD[2*NUM_C+NUM_C+2];
                c110_ = nodeCoefficientD[2*NUM_C+1+2];
                c111_ = nodeCoefficientD[2*NUM_C+NUM_C+1+2];

                c000_ = c000_ + h1.z*(c001_-c000_);
                c010_ = c010_ + h1.z*(c011_-c010_);
                c100_ = c100_ + h1.z*(c101_-c100_);
                c110_ = c110_ + h1.z*(c111_-c110_);

                c000_ = c000_ + h1.y*(c010_-c000_);
                c100_ = c100_ + h1.y*(c110_-c100_);

                double4_t c111;
                c111 =  c000_ + h1.x*(c100_-c000_);


                c000 = c001 + g0.z*(c000-c001);
                c010 = c011 + g0.z*(c010-c011);
                c100 = c101 + g0.z*(c100-c101);
                c110 = c111 + g0.z*(c110-c111);

                c000 = c010 + g0.y*(c000-c010);
                c100 = c110 + g0.y*(c100-c110);

                c000 = c100 + g0.x*(c000-c100);

                double4_t displacement=make_double4_t(0.0f,0.0f,0.0f,0.0f);
                displacement.x = c000.x;
                displacement.y = c000.y;
                displacement.z = c000.z;

                int tmp_index = z * referenceImageDim.y * referenceImageDim.x + y * referenceImageDim.x + x;

                positionField[tmp_index] = displacement;
            }
        }
    }

    return;
}
