#include "graphColoringPreprocess.h"
#include "deviceConsVars.h"
#include "cudaError.h"
#include "./quickSort/cdpSimpleQuicksort.h"
int * faceConflict;
int * posiFaceConflict;
int numFaceGroups;
int * faceGroup;
int * faceGroupNums;
int * posiFaceGroup;
int * d_faceGroup;
int * d_faceGroupOpt;
void graphColoringPreprocess(int numVolumes, int numFaces, const int * cellFaces, const int * posiCellFaces, int * owner, int * neighbour){
	CreateFaceConflict(numVolumes, numFaces, cellFaces, posiCellFaces, owner, neighbour);
	ColorFaces(numFaces);
	validateFaceGroup(numFaces, numVolumes, numFaceGroups, faceGroup, posiFaceGroup, faceGroupNums, owner, neighbour);
	hToDFaceGroupInfoTransfer(numFaces, faceGroup);
	initVolumeRenumberNew(numVolumes);
	cellRenumberByfaceGroup(numFaces, numVolumes, numFaceGroups, faceGroup, posiFaceGroup, faceGroupNums, owner, neighbour);
	hToDMeshOptInforTransfer(numFaces, owner, neighbour);
	validateFaceGroup(numFaces, numVolumes, numFaceGroups, faceGroup, posiFaceGroup, faceGroupNums, owner, neighbour);
	hToDFaceGroupOptInfoTransfer(numFaces, faceGroup);
	freeHostMemory(faceConflict, posiFaceConflict, faceGroup);
}

void CreateFaceConflict(int numVolumes, int numFaces, const int * cellFaces, const int * posiCellFaces, const int * owner, const int * neighbour){
	int faceID, ownFaceID, ngbFaceID, refFaceID;
	int sumFaceConflict = 0;
	posiFaceConflict = (int *)malloc(numFaces * sizeof(int));
	for (faceID = 0; faceID < numFaces; faceID++){
		int ownVolumeID = owner[faceID];
		int ngbVolumeID = neighbour[faceID];
		posiFaceConflict[faceID] = sumFaceConflict;
		int ownCellFaceID = posiCellFaces[ownVolumeID];
		sumFaceConflict += cellFaces[ownCellFaceID] -1;
		int ngbCellFaceID = posiCellFaces[ngbVolumeID];
		sumFaceConflict += cellFaces[ngbCellFaceID] -1;	
		sumFaceConflict++; //for store the number of conflict faces for one face
	}
	faceConflict = (int *)malloc(sumFaceConflict*sizeof(int));
	for (faceID = 0; faceID < numFaces; faceID++){
		int faceConflictID = posiFaceConflict[faceID];
		faceConflict[faceConflictID] = 0; //store the number of conflict faces of one face, initialize as zero
		int offsetFaceConflict = 0;
		int ownVolumeID = owner[faceID];
		int ownCellFaceID = posiCellFaces[ownVolumeID];
		int numOwnVolumeFaces = cellFaces[ownCellFaceID]; //The first element is the number of faces in one cell.
		for (refFaceID = 1; refFaceID < numOwnVolumeFaces + 1; refFaceID++) {
			ownFaceID = cellFaces[ownCellFaceID + refFaceID];	
			if (ownFaceID != faceID){
				offsetFaceConflict++;
				faceConflict[faceConflictID]++;
				faceConflict[faceConflictID + offsetFaceConflict] = ownFaceID;
			}
		}
		if (offsetFaceConflict != numOwnVolumeFaces -1) {
			printf("Something wrong in creating face conflict relation for own, offsetFaceConflict = %d, numOwnVolumeFaces = %d\n", offsetFaceConflict, numOwnVolumeFaces);
			printf("faceID = %d, ownVolumeID = %d,\n",faceID, ownVolumeID);
			int i = 0;
			for (i = 1; i < numOwnVolumeFaces + 1; i++){
				int ownFaceID = cellFaces[ownCellFaceID + i];	
				printf("on %d term, faceID = %d\n", i, ownFaceID);
			}
			exit(0);
		}

		int ngbVolumeID = neighbour[faceID];
		int ngbCellFaceID = posiCellFaces[ngbVolumeID];
		int numNgbVolumeFaces = cellFaces[ngbCellFaceID]; //The first element is the number of faces in one cell.
		for (refFaceID = 1; refFaceID < numNgbVolumeFaces + 1; refFaceID++) {
			ngbFaceID = cellFaces[ngbCellFaceID + refFaceID];	
			if (ngbFaceID != faceID){
					offsetFaceConflict++;
					faceConflict[faceConflictID]++;
					faceConflict[faceConflictID + offsetFaceConflict] = ngbFaceID;
			}
		}
		if (offsetFaceConflict != numOwnVolumeFaces -1 + numNgbVolumeFaces - 1) {
			printf("Something wrong in creating face conflict relation for own, offsetFaceConflict = %d, numOwnVolumeFaces = %d, numNgbVolumeFaces = %d\n", offsetFaceConflict, numOwnVolumeFaces, numNgbVolumeFaces);
			printf("faceID = %d, ownVolumeID = %d, ngbVolumeID = %d\n",faceID, ownVolumeID, ngbVolumeID);
			printf("ownCellFaceID = %d, ngbCellFaceID = %d\n", ownCellFaceID, ngbCellFaceID);
			int i = 0;
			for (i = 1; i < numOwnVolumeFaces + 1; i++){
				int ownFaceID = cellFaces[ownCellFaceID + i];	
				printf("on %d term, faceID = %d\n", i, ownFaceID);
			}
			for (i = 1; i < numNgbVolumeFaces + 1; i++){
				int ngbFaceID = cellFaces[ngbCellFaceID + i];	
				printf("on %d term, faceID = %d\n", i, ngbFaceID);
			}
			exit(0);
		}

	}
	//Test results of faceConflict and posiFaceConflict	
	for (faceID = 0; faceID < numFaces; faceID++){
		int faceConflictID = posiFaceConflict[faceID];
		int numFaceConflict = faceConflict[faceConflictID];
		int offsetFaceConflict = 0;
		for(offsetFaceConflict = 1; offsetFaceConflict < numFaceConflict+1; offsetFaceConflict++){
			if (faceID == faceConflict[faceConflictID + offsetFaceConflict]) {
				printf("Something wrong in faceConflict and posiFaceConflict for face %d\n", faceID);
				int i = 0;
				for (i = 1; i < numFaceConflict+1; i++) printf("face %d\n", faceConflict[faceConflictID + i]);
				exit(0);
			}
		}
	}
	printf("CreateFaceConflict successfully\n");
}
	
void ColorFaces(int numFaces){
	int faceID, colorID;
	int colorMax = 0;
	numFaceGroups = 0;
	faceGroup = (int *)malloc(numFaces * sizeof(int));
	int * faceColor = (int *)malloc(numFaces * sizeof(int));
	for (faceID = 0; faceID < numFaces; faceID++){
		faceColor[faceID] = -1;
	}

	for (faceID = 0; faceID < numFaces; faceID++){
		int color = 0;
		int colorSame = 0;	
		int faceConflictID = posiFaceConflict[faceID];
		int numFaceConflict = faceConflict[faceConflictID];
		int offsetFaceConflict = 0;
		while (faceColor[faceID] == -1){
			for(offsetFaceConflict = 1; offsetFaceConflict < numFaceConflict+1; offsetFaceConflict++){
				int conflictFaceID = faceConflict[faceConflictID + offsetFaceConflict];
				if (color == faceColor[conflictFaceID]) {
					colorSame = 1;
					break;
				}
			}
			if (colorSame == 0) faceColor[faceID]=color;
			else{
				color++;
				colorSame = 0;
			}
		}
		//record the maximum face color
		if (faceColor[faceID] > colorMax) colorMax = faceColor[faceID];
	}
	
	numFaceGroups = colorMax + 1;
	printf("After graph coloring, faces own %d colors\n", numFaceGroups);
	faceGroupNums = (int *)malloc(numFaceGroups*sizeof(int));
	posiFaceGroup = (int *)malloc(numFaceGroups*sizeof(int));
	int * offsetFaceColor = (int *)malloc(numFaceGroups*sizeof(int));
	posiFaceGroup[0] = 0;
	//initialization
	for (colorID = 0; colorID < numFaceGroups; colorID++){
		faceGroupNums[colorID] = 0;
		offsetFaceColor[colorID] = 0;
	}
	//calculate faceGroupNums
	for (faceID = 0; faceID < numFaces; faceID++){
		int color = faceColor[faceID];
		faceGroupNums[color]++;
	}
	//test and output faceGroupNums
	int numFacesByColor = 0;
	for (colorID = 0; colorID < numFaceGroups; colorID++){
		printf("On color %d, faceGroupNums[%d] = %d\n", colorID, colorID, faceGroupNums[colorID]);
		numFacesByColor += faceGroupNums[colorID];
	}
	if (numFacesByColor != numFaces) {
		printf("Something wrong in graph coloring, numFacesByColor = %d, numFaces=%d\n", numFacesByColor, numFaces);
		exit(0);
	}
	printf("Graph color successfully\n");
	//calculate posiFaceGroup
	for (colorID = 1; colorID < numFaceGroups; colorID++){
		posiFaceGroup[colorID] = posiFaceGroup[colorID - 1] + faceGroupNums[colorID - 1];
	}
	//set faceID into faceGroup
	for (faceID = 0; faceID < numFaces; faceID++){
		int colorID = faceColor[faceID];
		int posiFace = posiFaceGroup[colorID] + offsetFaceColor[colorID];
		faceGroup[posiFace] = faceID;
		offsetFaceColor[colorID]++;
	}
	//free host memory
	free(faceColor);
	//free(posiFaceGroup);
	free(offsetFaceColor);
}

void hToDFaceGroupInfoTransfer(int numFaces, const int * h_faceGroup){
	size_t faceSize = numFaces * sizeof(int);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_faceGroup, faceSize));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_faceGroup, h_faceGroup, faceSize, cudaMemcpyHostToDevice));
}

void hToDFaceGroupOptInfoTransfer(int numFaces, const int * h_faceGroup){
	size_t faceSize = numFaces * sizeof(int);
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_faceGroupOpt, faceSize));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_faceGroupOpt, h_faceGroup, faceSize, cudaMemcpyHostToDevice));
}

void freeHostMemory(int * faceConflict, int * posiFaceConflict, int * faceGroup){
	free(faceConflict);
	free(posiFaceConflict);
	free(faceGroup);
}
void freeMemoryGraphColoring(){
	HANDLE_CUDA_ERROR(cudaFree(d_faceGroup));
	HANDLE_CUDA_ERROR(cudaFree(d_faceGroupOpt));
	free(posiFaceGroup);
}

void faceGroupRenumberByOwner(int numFaceGroups, const int * faceGroupNums, int * faceGroup, const int * posiFaceGroup, const int * owner){
	int faceGroupID, posiFaceGroupID, numFaces;
	int i, j;
	int faceID0, faceID1, ownVolumeID0, ownVolumeID1;
	for (faceGroupID = 0; faceGroupID < numFaceGroups; faceGroupID++){
		posiFaceGroupID = posiFaceGroup[faceGroupID];
		numFaces = faceGroupNums[faceGroupID];
		for (i = 0; i < numFaces; i++){
			for (j = 0; j < numFaces - i - 1; j++){
				faceID0 = faceGroup[posiFaceGroupID+j];
				faceID1 = faceGroup[posiFaceGroupID+j+1];
				ownVolumeID0 = owner[faceID0];
				ownVolumeID1 = owner[faceID1];
				if (ownVolumeID0 > ownVolumeID1) {
					//exchange faceID0 and faceID1 in faceGroup
					faceGroup[posiFaceGroupID+j] = faceID1;
					faceGroup[posiFaceGroupID+j+1] = faceID0;	
				}
			}
		}
		//check the renumber results
		for (i = 0; i < numFaces-1; i++){
			faceID0 = faceGroup[posiFaceGroupID+i];
			faceID1 = faceGroup[posiFaceGroupID+i+1];
			ownVolumeID0 = owner[faceID0];
			ownVolumeID1 = owner[faceID1];
			
			if (ownVolumeID0 > ownVolumeID1) {
				printf("Error: faceID0>faceID1, faceGroupID = %d, faceID0 = %d, faceID1 = %d, ownVolumeID0 = %d, ownVolumeID1 = %d, posiFaceGroupID = %d, i = %d\n", faceGroupID, faceID0, faceID1, ownVolumeID0, ownVolumeID1, posiFaceGroupID, i);
				exit(0);
			}
		}
	}

}
void cellRenumberByfaceGroup(int numFaces, int numVolumes, int numFaceGroups, const int * faceGroup, const int * posiFaceGroup, const int * faceGroupNums, int * owner, int * neighbour){
	int faceGroupID, faceID, ownVolumeID, ngbVolumeID, volumeID, i, volumeIDNew;
	int faceGroupNumMaxID, numFacesInGroupMax, posiFaceMax;
	size_t sizeVolume = numVolumes * sizeof(int);
	//store whether or not a cell exists in the face group that owns the maximum faces
	int * volumeFlag = (int *)malloc(sizeVolume);
	//set volumeNumberNew with the old number, set flag with zero
	for (volumeID = 0; volumeID < numVolumes; volumeID++){
		//volumeNumberNew[volumeID] = volumeID;
		volumeFlag[volumeID] = 0;
	}
	faceGroupNumMaxID = 0;
	for (faceGroupID = 0; faceGroupID < numFaceGroups; faceGroupID++){
		if (faceGroupNums[faceGroupNumMaxID] < faceGroupNums[faceGroupID]) faceGroupNumMaxID = faceGroupID;
	}
	printf("faceGroup %d owns maximum faces %d\n", faceGroupNumMaxID, faceGroupNums[faceGroupNumMaxID]);
	numFacesInGroupMax = faceGroupNums[faceGroupNumMaxID];
	posiFaceMax = posiFaceGroup[faceGroupNumMaxID];
	for (i = 0; i < numFacesInGroupMax; i++){
		faceID = faceGroup[posiFaceMax + i];
		ownVolumeID = owner[faceID];
		ngbVolumeID = neighbour[faceID];
		//The cell ownVolumeID exists in the face group with maximum faces
		//The cell number must change from ownVolumeID into i
		volumeNumberNew[ownVolumeID] = i;
		volumeFlag[ownVolumeID] = 1;
		//volumeNumberNew[i] = ownVolumeID;
		//The cell ngbVolumeID exists in the face group with maximum faces
		//The cell number must change from ngbVolumeID into i + numFacesInGroupMax
		volumeNumberNew[ngbVolumeID] = i + numFacesInGroupMax;
		volumeFlag[ngbVolumeID] = 1;
	}
	//In fact, in this part, the other cell should be found and updated with new number
	for (volumeID = 0; volumeID < numVolumes; volumeID++){
		//update cell number by volumeFlag
		if (volumeFlag[volumeID] == 1) {
			volumeIDNew = volumeNumberNew[volumeID];
			int necessaryFlag = volumeFlag[volumeIDNew];
			//search the cell that does not exist in face group with maximum faces
			if (volumeIDNew != volumeID) {
				while(necessaryFlag) {
					//printf("volumeID = %d, volumeIDNew = %d\n", volumeID, volumeIDNew);
					volumeIDNew = volumeNumberNew[volumeIDNew];
					necessaryFlag = volumeFlag[volumeIDNew];
					if (volumeIDNew == volumeID) break;
				}
				if (volumeIDNew != volumeID) volumeNumberNew[volumeIDNew] = volumeID;
			}
		}
	}
	//update owner and neighbour by volmeNumberNew
	for (faceID = 0; faceID < numFaces; faceID++){
		ownVolumeID = owner[faceID];
		ngbVolumeID = neighbour[faceID];
		owner[faceID] = volumeNumberNew[ownVolumeID];
		neighbour[faceID] = volumeNumberNew[ngbVolumeID];
	}
	//check renumber result
	for (i = 0; i < numFacesInGroupMax; i++){
		faceID = faceGroup[posiFaceMax + i];
		ownVolumeID = owner[faceID];
		ngbVolumeID = neighbour[faceID];
		if (ownVolumeID != i) {
			printf("renumber fails: ownVolumeID = %d, i = %d\n", ownVolumeID, i);
			exit(1);
		}
		if (ngbVolumeID != i + numFacesInGroupMax) {
			printf("renumber fails: ngbVolumeID = %d, i + numFacesInGroupMax= %d\n", ngbVolumeID, i+numFacesInGroupMax);
			exit(1);
		}
     	} 
	printf("Test renumber successfully\n");
	free(volumeFlag);
}

void validateFaceGroup(int numFaces, int numVolumes, int numFaceGroups, const int * faceGroup, const int * posiFaceGroup, const int * faceGroupNums, const int * owner, const int * neighbour){
	int volumeID, faceGroupID, faceID, ownVolumeID, ngbVolumeID;
	int numFacesInGroup, posiFace, i;
	size_t sizeVolume = numVolumes * sizeof(int);
	int * volumeCount = (int *)malloc(sizeVolume);
	for (faceGroupID = 0; faceGroupID < numFaceGroups; faceGroupID++){
		for (volumeID = 0; volumeID < numVolumes; volumeID++){
			volumeCount[volumeID] = 0;
		}
		numFacesInGroup = faceGroupNums[faceGroupID];
		posiFace = posiFaceGroup[faceGroupID];
		for (i = 0; i < numFacesInGroup; i++){
			faceID = faceGroup[posiFace + i];
			ownVolumeID = owner[faceID];
			ngbVolumeID = neighbour[faceID];
			volumeCount[ownVolumeID]++;
			volumeCount[ngbVolumeID]++;
		}
		for (volumeID = 0; volumeID < numVolumes; volumeID++){
			if (volumeCount[volumeID] > 1) {
				printf("Test faceGroup fails: faceGroupID = %d, volumeCount[%d] = %d\n", faceGroupID, volumeID, volumeCount[volumeID]);
				exit(1);
			}
		}
	}
	printf("Test faceGroup successfully\n ");

}

void faceGroupRenumberByNeighbour(int numFaceGroups, const int * faceGroupNums, int * faceGroup, const int * posiFaceGroup, const int * neighbour){
	int faceGroupID, posiFaceGroupID, numFaces;
	int i, j;
	int faceID0, faceID1, ngbVolumeID0, ngbVolumeID1;
	for (faceGroupID = 0; faceGroupID < numFaceGroups; faceGroupID++){
		posiFaceGroupID = posiFaceGroup[faceGroupID];
		numFaces = faceGroupNums[faceGroupID];
		for (i = 0; i < numFaces; i++){
			for (j = 0; j < numFaces - i - 1; j++){
				faceID0 = faceGroup[posiFaceGroupID+j];
				faceID1 = faceGroup[posiFaceGroupID+j+1];
				ngbVolumeID0 = neighbour[faceID0];
				ngbVolumeID1 = neighbour[faceID1];
				if (ngbVolumeID0 > ngbVolumeID1) {
					//exchange faceID0 and faceID1 in faceGroup
					faceGroup[posiFaceGroupID+j] = faceID1;
					faceGroup[posiFaceGroupID+j+1] = faceID0;	
				}
			}
		}
		//check the renumber results
		for (i = 0; i < numFaces-1; i++){
			faceID0 = faceGroup[posiFaceGroupID+i];
			faceID1 = faceGroup[posiFaceGroupID+i+1];
			ngbVolumeID0 = neighbour[faceID0];
			ngbVolumeID1 = neighbour[faceID1];
			
			if (ngbVolumeID0 > ngbVolumeID1) {
				printf("Error: faceID0>faceID1, faceGroupID = %d, faceID0 = %d, faceID1 = %d, ngbVolumeID0 = %d, ngbVolumeID1 = %d, posiFaceGroupID = %d, i = %d\n", faceGroupID, faceID0, faceID1, ngbVolumeID0, ngbVolumeID1, posiFaceGroupID, i);
				exit(0);
			}
		}
	}

}

void faceGroupRenumberQuickSortOwner(int numFaceGroups, const int * faceGroupNums, int * faceGroup, const int * posiFaceGroup, const int * owner){
	int faceGroupID, ownVolumeID, faceID, ownVolumeIDAfter;
	int numFaces, posiFace;
	for (faceGroupID = 0; faceGroupID < numFaceGroups; faceGroupID++){
		printf("quick sort for %d face group by owner face number\n", faceGroupID);
		numFaces = faceGroupNums[faceGroupID];
		posiFace = posiFaceGroup[faceGroupID];
		int * ownVolumeIDArray = (int *)malloc(numFaces * sizeof(int));
		int * indexFaces = (int *)malloc(numFaces * sizeof(int));
		int * d_ownVolumeIDArray;
		int * d_indexFaces;
		size_t sizeFace = numFaces * sizeof(int);
		for (int i = 0; i < numFaces; i++){
			faceID = faceGroup[posiFace + i];
			ownVolumeID = owner[faceID];
			ownVolumeIDArray[i] = ownVolumeID;
			indexFaces[i] = faceID;
		}
		HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_ownVolumeIDArray, sizeFace));
		HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_indexFaces, sizeFace));
		HANDLE_CUDA_ERROR(cudaMemcpy(d_ownVolumeIDArray, ownVolumeIDArray, sizeFace, cudaMemcpyHostToDevice));
		HANDLE_CUDA_ERROR(cudaMemcpy(d_indexFaces, indexFaces, sizeFace, cudaMemcpyHostToDevice));
		run_qsort(d_ownVolumeIDArray, d_indexFaces, numFaces);
		//check the results of quick sort
		HANDLE_CUDA_ERROR(cudaMemcpy(ownVolumeIDArray, d_ownVolumeIDArray, sizeFace, cudaMemcpyDeviceToHost));
		HANDLE_CUDA_ERROR(cudaMemcpy(indexFaces, d_indexFaces, sizeFace, cudaMemcpyDeviceToHost));
		
		for (int i = 0; i < numFaces; i++){
			ownVolumeIDAfter = ownVolumeIDArray[i];
			faceID = indexFaces[i];
			ownVolumeID = owner[faceID];
			if (ownVolumeIDAfter != ownVolumeID) {
				printf("index quick sort fails in %d term: ownVolumeID = %d, ownVolumeIDAfter = %d, indexFaces[%d] = %d\n", i, ownVolumeID, ownVolumeIDAfter, i, indexFaces[i]);
				exit(0);
			}
			if (i != numFaces - 1) {
				if (ownVolumeIDArray[i] > ownVolumeIDArray[i+1]) {
					printf("owner face quick sort fails in %d term: ownVolumeIDArray[%d] = %d, ownVolumeIDArray[%d] = %d\n", i, i, ownVolumeIDArray[i], i+1, ownVolumeIDArray[i+1]);
					exit(0);
				}
			}
			
		}
		printf("Test quick sort successfully\n");
		//rest faceID by new order
		for (int i = 0; i< numFaces; i++) faceGroup[posiFace + i] = indexFaces[i];
		free(ownVolumeIDArray);
		free(indexFaces);
		HANDLE_CUDA_ERROR(cudaFree(d_ownVolumeIDArray));
		HANDLE_CUDA_ERROR(cudaFree(d_indexFaces));
	}

}
void initVolumeRenumberNew(int numVolumes){
	int volumeID;
	size_t sizeVolume = numVolumes * sizeof(int);
	volumeNumberNew = (int *)malloc(sizeVolume);
	for (volumeID = 0; volumeID < numVolumes; volumeID++){
		volumeNumberNew[volumeID] = volumeID;
	}
}
