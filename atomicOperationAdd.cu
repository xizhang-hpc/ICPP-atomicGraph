#include "atomicOperationAdd.h"
#include "atomicOperationMax.h"
#include "deviceConsVars.h"
#include "cudaError.h"
#include "Timer.h"
#include "graphColoringAdd.h"
#include "graphColoringPreprocess.h"

void CallAtomicOperationAdd(int numFaces, int numVolumes)
{
	
	int blockSize = 0;
	int gridSizeVolumes = 0;
	int gridSizeFaces = 0;
for (int numLoops = 0; numLoops<LOOPNUM;numLoops++)
	{
		blockSize = 22 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		kernelInit<<<gridSizeVolumes, blockSize>>>(numVolumes, d_residual);
		HANDLE_KERNEL_ERROR();
		blockSize = 2 * 32;
		gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
		timeGPUZero(3);
		kernelAtomicOperationAdd<<<gridSizeFaces, blockSize>>>(numFaces, d_owner, d_neighbour, d_flux, d_residual);
		timeGPUOne(3);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResults(numLoops, numVolumes, d_residual, residual, 13);
		blockSize = 24 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		kernelInit<<<gridSizeVolumes, blockSize>>>(numVolumes, d_residual);
		HANDLE_KERNEL_ERROR();
		blockSize = 2 * 32;
		gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
		timeGPUZero(5);
		//dynamic shared memory
		kernelAtomicOperationAddOpt<<<gridSizeFaces, blockSize, blockSize*sizeof(fpkind)>>>(numFaces, d_owner, d_neighbour, d_flux, d_residual);
		timeGPUOne(5);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResults(numLoops, numVolumes, d_residual, residual, 15);
	}
}
__global__ void kernelAtomicOperationAdd(const int numFaces, const int * owner, 
		const int * neighbour, const fpkind * flux, fpkind * residual)
{
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceID = 0;
        int ownVolumeID, ngbVolumeID;
        for (faceID = threadID; faceID < numFaces; faceID += gridDim.x * blockDim.x) 
	{
                ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		#if __CUDA_ARCH__ < 600
			atomicAddTest(residual+ownVolumeID, -flux[faceID]);
			atomicAddTest(residual+ngbVolumeID,  flux[faceID]);
		#else
			atomicAdd(residual+ownVolumeID, -flux[faceID]);
			atomicAdd(residual+ngbVolumeID,  flux[faceID]);
		#endif
	}
}

__global__ void kernelAtomicOperationAddOpt(const int numFaces, const int * owner, 
		const int * neighbour, const fpkind * flux, fpkind * residual)
{
	extern __shared__ fpkind fluxShare[];
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int threadLocalID = threadIdx.x;
	int faceID = 0;
        int ownVolumeID, ngbVolumeID;
        for (faceID = threadID; faceID < numFaces; faceID += gridDim.x * blockDim.x) 
	{
                ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		fluxShare[threadLocalID] = flux[faceID];
		#if __CUDA_ARCH__ < 600
			atomicAddTest(residual+ownVolumeID, -fluxShare[threadLocalID]);
			atomicAddTest(residual+ngbVolumeID,  fluxShare[threadLocalID]);
		#else
			atomicAdd(residual+ownVolumeID, -fluxShare[threadLocalID]);
			atomicAdd(residual+ngbVolumeID,  fluxShare[threadLocalID]);
		#endif
	}
}
#if __CUDA_ARCH__ < 600
__device__ double atomicAddTest(double* address, double val)
{
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
}

__device__ float atomicAddTest(float* address, float val)
{
        unsigned int* address_as_ull = (unsigned int*)address;
        unsigned int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,__float_as_uint(val +
__uint_as_float(assumed)));
        } while (assumed != old);
        return __uint_as_float(old);
}
#endif
