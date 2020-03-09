#include "graphColoringMax.h"
#include "graphColoringAdd.h"
#include "deviceConsVars.h"
#include "graphColoringPreprocess.h"
#include "cudaError.h"
#include "Timer.h"
void CallGraphColoringAdd(int numVolumes)
{
	int groupID;
	int numLoops;
	int blockSize = 0;
	int gridSizeVolumes = 0;
	int gridSizeFaces = 0;
	for (numLoops = 0; numLoops<LOOPNUM;numLoops++)
	{
		blockSize = 24 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		kernelInitGraphColoring<<<gridSizeVolumes, blockSize>>>(numVolumes, d_residual);
		HANDLE_KERNEL_ERROR();
		blockSize = 32;
		timeGPUZero(7);
		for (groupID = 0; groupID < numFaceGroups; groupID++) 
		{
			int numFaces = faceGroupNums[groupID];
			int posiGroup = posiFaceGroup[groupID];
			gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
			kernelGraphColoringAdd<<<gridSizeFaces, blockSize>>>(numFaces, posiGroup, d_faceGroup, 
						d_owner, d_neighbour, d_flux, d_residual);
		}
		timeGPUOne(7);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResults(numLoops, numVolumes, d_residual, residual, 19);
		blockSize = 9 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		kernelInitGraphColoring<<<gridSizeVolumes, blockSize>>>(numVolumes, d_residual);
		HANDLE_KERNEL_ERROR();
		blockSize = 32;
		timeGPUZero(13);
		for (groupID = 0; groupID < numFaceGroups; groupID++) 
		{
			int numFaces = faceGroupNums[groupID];
			int posiGroup = posiFaceGroup[groupID];
			gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
			kernelGraphColoringAddFluxOpt<<<gridSizeFaces, blockSize>>>(numFaces, posiGroup, d_faceGroup, d_owner, d_neighbour, d_fluxRe, d_residual);
		}
		timeGPUOne(13);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResults(numLoops, numVolumes, d_residual, residual, 21);
		blockSize = 32 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		kernelInitGraphColoring<<<gridSizeVolumes, blockSize>>>(numVolumes, d_residualRe);
		HANDLE_KERNEL_ERROR();
		blockSize = 32;
		timeGPUZero(11);
		for (groupID = 0; groupID < numFaceGroups; groupID++) 
		{
			int numFaces = faceGroupNums[groupID];
			int posiGroup = posiFaceGroup[groupID];
			gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
			kernelGraphColoringAdd<<<gridSizeFaces, blockSize>>>(numFaces, posiGroup, d_faceGroupOpt, 
						d_ownerOpt, d_neighbourOpt, d_flux, d_residualRe);
		}
		timeGPUOne(11);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResultsVolumeRenumber(numLoops, numVolumes, d_residualRe, residual, 23);
		blockSize = 27 * 32;
		gridSizeVolumes = (numVolumes + blockSize - 1) / blockSize;
		kernelInitGraphColoring<<<gridSizeVolumes, blockSize>>>(numVolumes, d_residual);
		HANDLE_KERNEL_ERROR();
		blockSize = 32;
		timeGPUZero(9);
		for (groupID = 0; groupID < numFaceGroups; groupID++) 
		{
			int numFaces = faceGroupNums[groupID];
			int posiGroup = posiFaceGroup[groupID];
			gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
			kernelGraphColoringAddShareOpt<<<gridSizeFaces, blockSize, blockSize*sizeof(fpkind)>>>(numFaces, posiGroup, d_faceGroup, d_owner, d_neighbour, d_flux, d_residual);
		}
		timeGPUOne(9);
		HANDLE_KERNEL_ERROR();
		validateHostDeviceResults(numLoops, numVolumes, d_residual, residual, 19);
	}

}

__global__ void kernelGraphColoringAdd(int numFaces, int posiGroup, const int * faceGroup, 
		const int * owner, const int * neighbour, const fpkind * flux, fpkind * residual)
{
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceGroupID, faceID;
	int ownVolumeID, ngbVolumeID;
	for (faceGroupID = threadID; faceGroupID < numFaces; faceGroupID += blockDim.x * gridDim.x){
		faceID = faceGroup[faceGroupID + posiGroup];
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		residual[ownVolumeID] -= flux[faceID];
                residual[ngbVolumeID] += flux[faceID];
	}

}
__global__ void kernelGraphColoringAddShareOpt(int numFaces, int posiGroup, const int * faceGroup, 
		const int * owner, const int * neighbour, const fpkind * flux, fpkind * residual)
{
	extern __shared__ fpkind fluxShare[];
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int threadLocalID = threadIdx.x;
	int faceGroupID, faceID;
	int ownVolumeID, ngbVolumeID;
	for (faceGroupID = threadID; faceGroupID < numFaces; faceGroupID += blockDim.x * gridDim.x){
		faceID = faceGroup[faceGroupID + posiGroup];
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		fluxShare[threadLocalID] = flux[faceID];
		residual[ownVolumeID] -= fluxShare[threadLocalID];
                residual[ngbVolumeID] += fluxShare[threadLocalID];
	}

}

__global__ void kernelGraphColoringAddFluxOpt(int numFaces, int posiGroup, const int * faceGroup, 
		const int * owner, const int * neighbour, const fpkind * flux, fpkind * residual)
{
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceGroupID, faceID;
	int ownVolumeID, ngbVolumeID;
	for (faceGroupID = threadID; faceGroupID < numFaces; faceGroupID += blockDim.x * gridDim.x){
		faceID = faceGroup[faceGroupID + posiGroup];
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		residual[ownVolumeID] -= flux[faceGroupID + posiGroup];
                residual[ngbVolumeID] += flux[faceGroupID + posiGroup];
	}

}

__global__ void kernelGraphColoringAddFluxOptShareOpt(int numFaces, int posiGroup, const int * faceGroup, 
		const int * owner, const int * neighbour, const fpkind * flux, fpkind * residual)
{
	extern __shared__ fpkind fluxShare[];
	int threadLocalID = threadIdx.x;
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int faceGroupID, faceID;
	int ownVolumeID, ngbVolumeID;
	for (faceGroupID = threadID; faceGroupID < numFaces; faceGroupID += blockDim.x * gridDim.x){
		faceID = faceGroup[faceGroupID + posiGroup];
		ownVolumeID = owner[faceID];
                ngbVolumeID = neighbour[faceID];
		fluxShare[threadLocalID] = flux[faceGroupID + posiGroup];
		residual[ownVolumeID] -= fluxShare[threadLocalID];
                residual[ngbVolumeID] += fluxShare[threadLocalID];
	}

}

