#include "graphColoringMax.h"
#include "deviceConsVars.h"
#include "graphColoringPreprocess.h"
#include "cudaError.h"
#include "Timer.h"
void CallGraphColoringMax(int numVolumes){
	int groupID;
	int numLoops;
	int blockSize = 0;
	int gridSizeVolumes = 0;
	int gridSizeFaces = 0;
for (numLoops = 0; numLoops<LOOPNUM;numLoops++){
	blockSize = 13 * 32;
	gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
	kernelInitGraphColoring<<<gridSizeVolumes, blockSize>>>(numVolumes, d_pressureLocalMax);
        HANDLE_KERNEL_ERROR();
	blockSize = 32;
	timeGPUZero(8);
	for (groupID = 0; groupID < numFaceGroups; groupID++) {
		int numFaces = faceGroupNums[groupID];
		int posiGroup = posiFaceGroup[groupID];
		gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
		kernelGraphColoringMax<<<gridSizeFaces, blockSize>>>(numFaces, posiGroup, d_faceGroup, d_owner, d_neighbour, d_pressure, d_pressureLocalMax);
	}
	timeGPUOne(8);
        HANDLE_KERNEL_ERROR();
	validateHostDeviceResults(numLoops, numVolumes, d_pressureLocalMax, pressureLocalMax, 5);
	blockSize = 25 * 32;
	gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
	kernelInitGraphColoring<<<gridSizeVolumes, blockSize>>>(numVolumes, d_pressureLocalMax);
        HANDLE_KERNEL_ERROR();
	blockSize = 32;
	timeGPUZero(12);
	for (groupID = 0; groupID < numFaceGroups; groupID++) {
		int numFaces = faceGroupNums[groupID];
		int posiGroup = posiFaceGroup[groupID];
		gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
		kernelGraphColoringMax<<<gridSizeFaces, blockSize>>>(numFaces, posiGroup, d_faceGroupOpt, d_ownerOpt, d_neighbourOpt, d_pressureRe, d_pressureLocalMax);
	}
	timeGPUOne(12);
        HANDLE_KERNEL_ERROR();
	validateHostDeviceResultsVolumeRenumber(numLoops, numVolumes, d_pressureLocalMax, pressureLocalMax, 7);
	blockSize = 14 * 32;
	gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
	kernelInitGraphColoring<<<gridSizeVolumes, blockSize>>>(numVolumes, d_pressureLocalMax);
        HANDLE_KERNEL_ERROR();
	blockSize = 32;
	timeGPUZero(10);
	for (groupID = 0; groupID < numFaceGroups; groupID++) {
		int numFaces = faceGroupNums[groupID];
		int posiGroup = posiFaceGroup[groupID];
		gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
		kernelGraphColoringMaxOpt<<<gridSizeFaces, blockSize, blockSize*2*sizeof(fpkind)>>>(numFaces, posiGroup, d_faceGroup, d_owner, d_neighbour, d_pressure, d_pressureLocalMax);
	}
	timeGPUOne(10);
        HANDLE_KERNEL_ERROR();
	validateHostDeviceResults(numLoops, numVolumes, d_pressureLocalMax, pressureLocalMax, 11);
}

}

__global__ void kernelInitGraphColoring(int numVolumes, fpkind  * pressureLocalMax){
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
        int volumeID = 0;
        for (volumeID = threadID; volumeID < numVolumes; volumeID += gridDim.x * blockDim.x) {
                pressureLocalMax[volumeID] = 0.0;
        }
}

__global__ void kernelGraphColoringMax(int numFaces, int posiGroup, const int * faceGroup, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax){
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceGroupID, faceID;
	int ownVolumeID, ngbVolumeID;
	for (faceGroupID = threadID; faceGroupID < numFaces; faceGroupID += blockDim.x * gridDim.x){
		faceID = faceGroup[faceGroupID + posiGroup];
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		pressureLocalMax[ownVolumeID] = kernelMAXDOUBLEGraph(pressureLocalMax[ownVolumeID], pressure[ngbVolumeID]);
                pressureLocalMax[ngbVolumeID] = kernelMAXDOUBLEGraph(pressureLocalMax[ngbVolumeID], pressure[ownVolumeID]);
	}

}

__global__ void kernelGraphColoringMaxOpt(int numFaces, int posiGroup, const int * faceGroup, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax){
	extern __shared__ fpkind pressureNgbOwn[];
	int threadLocalID = threadIdx.x;
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceGroupID, faceID;
	int ownVolumeID, ngbVolumeID;
	for (faceGroupID = threadID; faceGroupID < numFaces; faceGroupID += blockDim.x * gridDim.x){
		faceID = faceGroup[faceGroupID + posiGroup];
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		pressureNgbOwn[2*threadLocalID+0] = pressure[ngbVolumeID];
                pressureNgbOwn[2*threadLocalID+1] = pressure[ownVolumeID];
		pressureLocalMax[ownVolumeID] = kernelMAXDOUBLEGraph(pressureLocalMax[ownVolumeID], pressureNgbOwn[2*threadLocalID+0]);
                pressureLocalMax[ngbVolumeID] = kernelMAXDOUBLEGraph(pressureLocalMax[ngbVolumeID], pressureNgbOwn[2*threadLocalID+1]);
	}

}

__device__ fpkind kernelMAXDOUBLEGraph(fpkind a, fpkind b){
	return ((a>b)?a:b);
}
