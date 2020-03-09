#include "stdio.h"
#include "precision.h"
void CallAtomicOperationMax(int numFaces, int numVolumes);
__global__ void kernelInit(int numVolumes, fpkind * pressureLocalMax);
__global__ void kernelAtomicOperationMax(const int numFaces, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax);
__device__ void atomicMax(double *addr, double val);
__device__ void atomicMax(float *addr, float val);
__device__ fpkind kernelMAXDOUBLE(fpkind a, fpkind b);
__global__ void kernelAtomicOperationMaxOpt(const int numFaces, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax);
