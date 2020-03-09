#include "validation.h"
#include "deviceConsVars.h"
#include "cudaError.h"
void ValidateAtomicOperationMax(int numVolumes, const fpkind * hostPressureLocalMax){
	int volumeID;
	size_t volumeSize = numVolumes * sizeof(fpkind);
	fpkind * devicePressureLocalMax = (fpkind *)malloc(volumeSize);
	HANDLE_CUDA_ERROR(cudaMemcpy(devicePressureLocalMax, d_pressureLocalMax, volumeSize, cudaMemcpyDeviceToHost));
	double errorMax = 1.0e-40;
	int  maxVolumeID = -1;
	for (volumeID = 0; volumeID < numVolumes; volumeID ++){
		double error = fabs(devicePressureLocalMax[volumeID]-hostPressureLocalMax[volumeID]);
		if (error > errorMax){
			errorMax = error;
			maxVolumeID = volumeID;
		}
	}
	if (maxVolumeID == -1) printf("Comparing pressureLoadMax in device and host, the maximum error is %.30e\n in volumeID %d\n", errorMax, maxVolumeID);
	else printf("Comparing pressureLoadMax in device and host, the maximum error is %.30e and the reference error is %.30e  in volumeID %d\n", errorMax, errorMax/(fabs(hostPressureLocalMax[maxVolumeID])+1.0e-40), maxVolumeID);
	free(devicePressureLocalMax);
}
void ValidateAtomicOperationAdd(int numVolumes, const fpkind * hostResidual){
	int volumeID;
	size_t volumeSize = numVolumes * sizeof(fpkind);
	fpkind * deviceResidual = (fpkind *)malloc(volumeSize);
	HANDLE_CUDA_ERROR(cudaMemcpy(deviceResidual, d_residual, volumeSize, cudaMemcpyDeviceToHost));
	double errorMax = 1.0e-40;
	int  maxVolumeID = -1;
	for (volumeID = 0; volumeID < numVolumes; volumeID ++){
		double error = fabs(deviceResidual[volumeID]-hostResidual[volumeID]);
		if (error > errorMax){
			errorMax = error;
			maxVolumeID = volumeID;
		}
	}
	if (maxVolumeID == -1) printf("Comparing residual in device and host, the absolute maximum error is %.30e in volumeID %d\n", errorMax, maxVolumeID);
	else printf("Comparing residual in device and host, the absolute maximum error is %.30e and reference error i %.30e in volumeID %d\n", errorMax, errorMax/(fabs(hostResidual[maxVolumeID])), maxVolumeID);
	free(deviceResidual);
}
