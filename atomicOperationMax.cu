#include "atomicOperationMax.h"
#include "deviceConsVars.h"
#include "cudaError.h"
#include "Timer.h"
void CallAtomicOperationMax(int numFaces, int numVolumes){
	int numLoops;
	int blockSize = 0;
	int gridSizeVolumes = 0;
	int gridSizeFaces = 0;
	for (numLoops = 0; numLoops<LOOPNUM;numLoops++){
		blockSize = 26 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		timeGPUZero(0);
		kernelInit<<<gridSizeVolumes, blockSize>>>(numVolumes, d_pressureLocalMax);
		timeGPUOne(0);
		HANDLE_KERNEL_ERROR();
		blockSize = 13 * 32;
		gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
		timeGPUZero(6);
		kernelAtomicOperationMaxOpt<<<gridSizeFaces, blockSize, blockSize * 2*sizeof(fpkind)>>>(numFaces, d_owner, d_neighbour, d_pressure, d_pressureLocalMax);
		timeGPUOne(6);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResults(numLoops, numVolumes, d_pressureLocalMax, pressureLocalMax, 1);
		blockSize = 16 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		kernelInit<<<gridSizeVolumes, blockSize>>>(numVolumes, d_pressureLocalMax);
		HANDLE_KERNEL_ERROR();
		blockSize = 31 * 32;
		gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
		timeGPUZero(4);
		kernelAtomicOperationMax<<<gridSizeFaces, blockSize>>>(numFaces, d_owner, d_neighbour, d_pressure, d_pressureLocalMax);
		timeGPUOne(4);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResults(numLoops, numVolumes, d_pressureLocalMax, pressureLocalMax, 3);
	}

}
__global__ void kernelInit(int numVolumes, fpkind  * pressureLocalMax){
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int volumeID = 0;
	for (volumeID = threadID; volumeID < numVolumes; volumeID += gridDim.x * blockDim.x) {
		pressureLocalMax[volumeID] = 0.0;
	}
}

__global__ void kernelAtomicOperationMax(const int numFaces, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax){
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceID = 0;
	int ownVolumeID, ngbVolumeID;
	for (faceID = threadID; faceID < numFaces; faceID += gridDim.x * blockDim.x) {
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		atomicMax(pressureLocalMax + ownVolumeID, pressure[ngbVolumeID]);
		atomicMax(pressureLocalMax + ngbVolumeID, pressure[ownVolumeID]);
	}
}

__global__ void kernelAtomicOperationMaxOpt(const int numFaces, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax){
	extern __shared__ fpkind pressureNgbOwn[];
	int threadLocalID = threadIdx.x;
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceID = 0;
	int ownVolumeID, ngbVolumeID;
	for (faceID = threadID; faceID < numFaces; faceID += gridDim.x * blockDim.x) {
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		pressureNgbOwn[2*threadLocalID+0] = pressure[ngbVolumeID];
		pressureNgbOwn[2*threadLocalID+1] = pressure[ownVolumeID];

		atomicMax(pressureLocalMax + ownVolumeID, pressureNgbOwn[2*threadLocalID+0]);
		atomicMax(pressureLocalMax + ngbVolumeID, pressureNgbOwn[2*threadLocalID+1]);
	}
}
__device__ void atomicMax(double *addr, double val){
		unsigned long long int * addr_as_ull = (unsigned long long int *)(addr);
		unsigned long long int old = *addr_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(kernelMAXDOUBLE(val, __longlong_as_double(assumed))));
		} while (assumed != old);

}

__device__ void atomicMax(float *addr, float val){
		unsigned int * addr_as_ull = (unsigned int *)(addr);
		unsigned int old = *addr_as_ull, assumed;
		do{ 
			assumed = old;
			old = atomicCAS(addr_as_ull, assumed, __float_as_uint(kernelMAXDOUBLE(val, __uint_as_float(assumed))));
		} while (assumed != old);

}

__device__ fpkind kernelMAXDOUBLE(fpkind a, fpkind b){
	return ((a>b)?a:b);
}


