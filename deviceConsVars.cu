#include "stdio.h"
#include "deviceConsVars.h"
#include "graphColoringPreprocess.h"
#include "cuda_runtime.h"
#include "cudaError.h"
int * d_owner;
int * d_neighbour;
int * d_ownerOpt;
int * d_neighbourOpt;
int * volumeNumberNew;
fpkind * d_pressure;
fpkind * d_pressureRe;
fpkind * d_pressureLocalMax;
fpkind * d_flux;
fpkind * d_fluxRe;
fpkind * d_fluxReOpt;
fpkind * d_residual;
fpkind * d_residualRe;
fpkind * pressureLocalMax;
fpkind * residual;
int LOOPNUM = 1000;



void hToDMeshInforTransfer(int numFaces, const int * h_owner, const int * h_neighbour){
	size_t faceSize = numFaces * sizeof(int);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_owner, faceSize));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_neighbour, faceSize));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_owner, h_owner, faceSize, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_neighbour, h_neighbour, faceSize, cudaMemcpyHostToDevice));


}

void hToDMeshOptInforTransfer(int numFaces, const int * h_owner, const int * h_neighbour){
	size_t faceSize = numFaces * sizeof(int);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_ownerOpt, faceSize));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_neighbourOpt, faceSize));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_ownerOpt, h_owner, faceSize, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_neighbourOpt, h_neighbour, faceSize, cudaMemcpyHostToDevice));


}
void hToDConsInforTransfer(int numFaces, int numVolumes, const fpkind * h_pressure, const fpkind *h_flux)
{
	size_t volumeSize = numVolumes * sizeof(fpkind);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_pressure, volumeSize));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_pressure, h_pressure, volumeSize, cudaMemcpyHostToDevice));

	size_t fluxSize= numFaces * sizeof(fpkind);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_flux,fluxSize));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_flux, h_flux, fluxSize, cudaMemcpyHostToDevice));
}

void DeviceVarsMemAlloc(int numVolumes){
	size_t volumeSize = numVolumes * sizeof(fpkind);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_pressureLocalMax, volumeSize));

	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_residual, volumeSize));
}

void CallFaceVariablesRenumber(int numFaces, int numVolumes){	
	int faceGroupID, numFacesInGroup, posiFaceGroupID;
	size_t sizeFaces = numFaces * sizeof(fpkind);
	size_t sizeVolumes = numVolumes * sizeof(fpkind);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_fluxRe, sizeFaces));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_residualRe, sizeVolumes));
	for (faceGroupID = 0; faceGroupID < numFaceGroups; faceGroupID++){
		numFacesInGroup = faceGroupNums[faceGroupID];
		posiFaceGroupID = posiFaceGroup[faceGroupID];
		kernelFaceVariablesRenumber<<<1024, 1024>>>(numFacesInGroup, posiFaceGroupID, d_faceGroup, d_flux, d_fluxRe);
	}
	HANDLE_KERNEL_ERROR();
	printf("finish face variables renumber\n");
}

void CallFaceVariablesRenumberOpt(int numFaces, int numVolumes){	
	int faceGroupID, numFacesInGroup, posiFaceGroupID;
	size_t sizeFaces = numFaces * sizeof(fpkind);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_fluxReOpt, sizeFaces));
	for (faceGroupID = 0; faceGroupID < numFaceGroups; faceGroupID++){
		numFacesInGroup = faceGroupNums[faceGroupID];
		posiFaceGroupID = posiFaceGroup[faceGroupID];
		kernelFaceVariablesRenumber<<<1024, 1024>>>(numFacesInGroup, posiFaceGroupID, d_faceGroupOpt, d_flux, d_fluxReOpt);
	}
	HANDLE_KERNEL_ERROR();
	printf("finish face variables renumber in renumber faces\n");
}

void CallVolumeVariablesRenumber(int numVolumes){
	size_t sizeVolumes = numVolumes * sizeof(fpkind);
	size_t sizeVolumesINT = numVolumes * sizeof(int);
	int * d_volumeNumberNew;
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_pressureRe, sizeVolumes));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_volumeNumberNew, sizeVolumesINT));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_volumeNumberNew, volumeNumberNew, sizeVolumesINT, cudaMemcpyHostToDevice));
	kernelVolumeVariablesRenumber<<<1024, 1024>>>(numVolumes, d_volumeNumberNew, d_pressure, d_pressureRe);
	HANDLE_KERNEL_ERROR();
	HANDLE_CUDA_ERROR(cudaFree(d_volumeNumberNew));
	printf("finish volume variables renumber in renumber volumes\n");
}
__global__ void kernelVolumeVariablesRenumber(int numVolumes, const int * volumeNumberNew, const fpkind * pressure, fpkind * pressureRe){
	int volumeID = 0;
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;	
	for (volumeID = threadID; volumeID < numVolumes; volumeID += blockDim.x * gridDim.x){
		pressureRe[volumeNumberNew[volumeID]] = pressure[volumeID];
	}
}

__global__ void kernelFaceVariablesRenumber(int numFaces, int posiFace, const int * faceGroup, const fpkind * variablesOrg, fpkind * variablesOpt){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	int faceGroupID, faceID;
	for(faceGroupID = threadID; faceGroupID < numFaces; faceGroupID += blockDim.x*gridDim.x){
		faceID = faceGroup[posiFace + faceGroupID];
		variablesOpt[posiFace+faceGroupID] = variablesOrg[faceID];
	}
}
void cudaMemoryFree(){
	HANDLE_CUDA_ERROR(cudaFree(d_owner));
	HANDLE_CUDA_ERROR(cudaFree(d_neighbour));
	HANDLE_CUDA_ERROR(cudaFree(d_ownerOpt));
	HANDLE_CUDA_ERROR(cudaFree(d_neighbourOpt));
	HANDLE_CUDA_ERROR(cudaFree(d_pressure));
	HANDLE_CUDA_ERROR(cudaFree(d_pressureRe));
	HANDLE_CUDA_ERROR(cudaFree(d_pressureLocalMax));
	HANDLE_CUDA_ERROR(cudaFree(d_flux));
	HANDLE_CUDA_ERROR(cudaFree(d_fluxRe));
	HANDLE_CUDA_ERROR(cudaFree(d_residual));
	HANDLE_CUDA_ERROR(cudaFree(d_residualRe));
}

void  validateHostDeviceResults(int numLoops, int numElements, const fpkind * deviceValues, const fpkind * hostValues, int funcID){
	if (numLoops != 0) return;
	int elementID;
	size_t elementSize = numElements * sizeof(fpkind);
	fpkind * d_values = (fpkind*)malloc(elementSize);
	HANDLE_CUDA_ERROR(cudaMemcpy(d_values, deviceValues, elementSize, cudaMemcpyDeviceToHost));
        double errorMax = 1.0e-40;
        int  maxElementID = -1;
	for (elementID = 0; elementID < numElements; elementID ++){
                double error = fabs(d_values[elementID] - hostValues[elementID]);
                if (error > errorMax){
                        errorMax = error;
                        maxElementID = elementID;
                }
                //for check renumber result
        }
        if (maxElementID == -1) printf("In function %d, comparing pressureLoadMax in device and host, the maximum error is %.30e in volumeID %d\n", funcID, errorMax, maxElementID);
        else printf("In function %d, comparing pressureLoadMax in device and host, the maximum error is %.30e and the reference error is %.30e  in volumeID %d\n", funcID, errorMax, errorMax/(fabs(hostValues[maxElementID])+1.0e-40), maxElementID);
	free(d_values);
}
void  validateHostDeviceResultsVolumeRenumber(int numLoops, int numElements, const fpkind * deviceValues, const fpkind * hostValues, int funcID){
	if (numLoops != 0) return;
	int elementID, elementIDNew;
	size_t elementSize = numElements * sizeof(fpkind);
	fpkind * d_values = (fpkind*)malloc(elementSize);
	HANDLE_CUDA_ERROR(cudaMemcpy(d_values, deviceValues, elementSize, cudaMemcpyDeviceToHost));
        double errorMax = 1.0e-40;
        int  maxElementID = -1;
	for (elementID = 0; elementID < numElements; elementID ++){
		elementIDNew = volumeNumberNew[elementID];
                double error = fabs(d_values[elementIDNew] - hostValues[elementID]);
                if (error > errorMax){
                        errorMax = error;
                        maxElementID = elementID;
                }
                //for check renumber result
        }
        if (maxElementID == -1) printf("In function %d, comparing pressureLoadMax in device and host, the maximum error is %.30e in volumeID %d\n", funcID, errorMax, maxElementID);
        else printf("In function %d, comparing pressureLoadMax in device and host, the maximum error is %.30e and the reference error is %.30e  in volumeID %d\n", funcID, errorMax, errorMax/(fabs(hostValues[maxElementID])+1.0e-40), maxElementID);
	free(d_values);
}
