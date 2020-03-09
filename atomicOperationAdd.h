#ifndef _ATOMICADD_
#define _ATOMICADD_
#include <cuda.h>
#include "precision.h"

void CallAtomicOperationAdd(int numFaces, int numVolumes);
__global__ void kernelAtomicOperationAdd(const int numFaces, const int * owner, 
		const int * neighbour, const fpkind * flux, fpkind * residual);
__global__ void kernelAtomicOperationAddOpt(const int numFaces, const int * owner, 
		const int * neighbour, const fpkind * flux, fpkind * residual);
#if __CUDA_ARCH__ < 600
__device__ float atomicAddTest(float* address, float val);
__device__ double atomicAddTest(double* address, double val);
#endif

#endif
