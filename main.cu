#include "stdio.h"
#include "precision.h"
#include "malloc.h"
#include "deviceConsVars.h"
#include "deviceQuery.h"
#include "atomicOperationMax.h"
#include "atomicOperationAdd.h"
#include "validation.h"
#include "graphColoringPreprocess.h"
#include "graphColoringMax.h"
#include "graphColoringAdd.h"
#include "stdlib.h"
#include "time.h"
#include "Timer.h"
fpkind MAXDOUBLE(fpkind, fpkind);
fpkind RANDOMNUMBER(fpkind, fpkind, int);
int main(){
	printf("Performance comparison of Graph coloring and atomic operations for indirect memory access induced race conditoin on GPU in unstructured FVM CFD\n");
	deviceQuery();
	//files for reading mesh information
	/*declare*/
	FILE * ownerFile;
	FILE * neighbourFile;
	FILE * cellFacesFile;
	FILE * posiCellFacesFile;
	//The number of faces in mesh
	int numFaces = 0; 
	//The number of volumes in mesh
	int numVolumes = 0;
	//The number of data in cellFaces
	int numCellFaces = 0;
	int * owner;
	int * neighbour;
	//input variables for atomicMax and graphColoringMax
	fpkind * pressure;
	//ouput variables for atomicMax and graphColoringMax
	fpkind * flux;
	int faceID, volumeID;
	//Initialize the Timer
	timerGPUStart();
	ownerFile = fopen("owner", "r");
	neighbourFile = fopen("neighbour", "r");
	cellFacesFile = fopen("cellFaces", "r");
	posiCellFacesFile = fopen("posiCellFaces", "r");
	if ((ownerFile == NULL) || (neighbourFile == NULL)||(cellFacesFile == NULL)||(posiCellFacesFile == NULL)) {
		printf("Error: files open fails\n");
		//exit(0);
	}
	//read data in owner
	fread(&numFaces, sizeof(int), 1, ownerFile);
	owner = (int *)malloc(sizeof(int) * numFaces);
	fread(owner, sizeof(int), numFaces, ownerFile);
	//read data in neighbour
	fread(&numFaces, sizeof(int), 1, neighbourFile);
	neighbour = (int *)malloc(sizeof(int) * numFaces);
	fread(neighbour, sizeof(int), numFaces, neighbourFile);
	//read data in cellFaces
	fread(&numCellFaces, sizeof(int), 1, cellFacesFile);
	int * cellFaces = (int *)malloc(numCellFaces * sizeof(int));
	fread(cellFaces, sizeof(int), numCellFaces, cellFacesFile);
	//read data in posiCellFaces
	fread(&numVolumes, sizeof(int), 1, posiCellFacesFile);
	int * posiCellFaces = (int *)malloc(numVolumes * sizeof(int));
	fread(posiCellFaces, sizeof(int), numVolumes, posiCellFacesFile);
	pressure = (fpkind *)malloc(sizeof(fpkind) * numVolumes);
	pressureLocalMax = (fpkind *)malloc(sizeof(fpkind) * numVolumes);

	flux= (fpkind*)malloc(sizeof(fpkind)* numFaces);
	residual= (fpkind*)malloc(sizeof(fpkind)* numVolumes);
	srand(time(NULL));
	for (volumeID = 0; volumeID < numVolumes; volumeID++){
		pressure[volumeID] = RANDOMNUMBER(-1.0, 1.0, numVolumes);
	}
	printf("numFaces: %d\n", numFaces);
	printf("numVolumes: %d\n", numVolumes);
	//initialization of pressureLocalMax with zero (output)
	for (volumeID = 0; volumeID < numVolumes; volumeID++){
		pressureLocalMax[volumeID] = 0.0;
	}
	// flux initialization
	for(int faceID= 0; faceID< numFaces; faceID++)
	{
		flux[faceID]= RANDOMNUMBER(-1.0,0.1, numFaces);
	}
	// residual initialization
	for(volumeID = 0; volumeID < numVolumes; volumeID++)
	{
		residual[volumeID]= 0.0;
	}

	/* validation case */ 
	//Finding the local max pressure as validation
	for (faceID = 0; faceID < numFaces; faceID++) {
		int ownVolumeID = owner[faceID];
		int ngbVolumeID = neighbour[faceID];
		pressureLocalMax[ownVolumeID] = MAXDOUBLE(pressureLocalMax[ownVolumeID], pressure[ngbVolumeID]);
		pressureLocalMax[ngbVolumeID] = MAXDOUBLE(pressureLocalMax[ngbVolumeID], pressure[ownVolumeID]);
	}
	//Calculating the residual for validation
	for (faceID = 0; faceID < numFaces; faceID++) 
	{
                int ownVolumeID = owner[faceID];
                int ngbVolumeID = neighbour[faceID];
		residual[ownVolumeID] -=flux[faceID];
		residual[ngbVolumeID] +=flux[faceID];
        }

	//atomicMax 
	/*atomicMax case*/
	//transfer and allocate data for kernels
	hToDMeshInforTransfer(numFaces, owner, neighbour);
	hToDConsInforTransfer(numFaces, numVolumes, pressure, flux);
	DeviceVarsMemAlloc(numVolumes);
	/*graph color */
	//Preprocess
	graphColoringPreprocess(numVolumes, numFaces, cellFaces, posiCellFaces, owner, neighbour);
	//renumber variables on faces and volumes
	CallFaceVariablesRenumber(numFaces, numVolumes);
	CallFaceVariablesRenumberOpt(numFaces, numVolumes);
	CallVolumeVariablesRenumber(numVolumes);
	printf("finish faces and volumes renumber\n");
	// kernel conduct
	CallAtomicOperationMax(numFaces, numVolumes);
	// graph color kernel conduct
	CallGraphColoringMax(numVolumes);
	/*atomicAdd */
	// kernel conduct
	CallAtomicOperationAdd(numFaces, numVolumes);

	/*graph color for add operation*/
	CallGraphColoringAdd(numVolumes);
	

	timerGPUEnd();
	cudaMemoryFree();
	freeMemoryGraphColoring();
	free(owner);
	free(neighbour);
	free(pressure);
	free(cellFaces);
	free(posiCellFaces);
	free(volumeNumberNew);
	fclose(ownerFile);
	fclose(neighbourFile);
	fclose(cellFacesFile);
	fclose(posiCellFacesFile);
	return 0;
}
fpkind MAXDOUBLE(fpkind a, fpkind b){
	return ((a>b)?a:b);
}

fpkind RANDOMNUMBER(fpkind upper, fpkind lower, int numVolumes){
	if (numVolumes<=1){
		printf("Error: number of volumes is not larger than 1\n");
		exit(1);
	}
	//double x = ((double)rand()%numVolumes/(numVolumes-1));
	int x = rand()%numVolumes;
	return  lower + x*(upper-lower)/(numVolumes-1);
}
