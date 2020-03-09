#ifndef DEVICECONSVARS
#define DEVICECONSVARS
#include "precision.h"
extern int * d_owner;
extern int * d_neighbour;
extern int * d_ownerOpt; //optimize by cell renumber in faceGroup
extern int * d_neighbourOpt; //optimize by cell renumber in faceGroup
extern int * volumeNumberNew;
extern fpkind * d_pressure;
extern fpkind * d_pressureRe;
extern fpkind * d_pressureLocalMax;
extern fpkind * d_flux;
extern fpkind * d_fluxRe;
extern fpkind * d_fluxReOpt; //renumber with optimized face number
extern fpkind * d_residual;
extern fpkind * d_residualRe;
extern fpkind * pressureLocalMax;
extern fpkind * residual;
extern int LOOPNUM;


void hToDMeshInforTransfer(int numFaces, const int * owner, const int * neighbour);
void hToDMeshOptInforTransfer(int numFaces, const int * owner, const int * neighbour);
void hToDConsInforTransfer(int numFaces, int numVolumes, const fpkind * h_pressure, const fpkind *h_flux);
void DeviceVarsMemAlloc(int numVolumes);
void CallFaceVariablesRenumber(int numFaces, int numVolumes);
void CallFaceVariablesRenumberOpt(int numFaces, int numVolumes);
void CallVolumeVariablesRenumber(int numVolumes);
__global__ void kernelVolumeVariablesRenumber(int numVolumes, const int * d_volumeNumberNew, const fpkind * d_pressureLocalMax, fpkind * d_pressureLocalMaxRe);
__global__ void kernelFaceVariablesRenumber(int numFaces, int posiFace, const int * faceGroup, const fpkind * variablesOrg, fpkind * variablesOpt);
void compareHostDeviceValues(int numElement, const fpkind * deviceValues, const fpkind * hostValues, int funcID);
void validateHostDeviceResults(int numLoops, int numElements, const fpkind * deviceValues, const fpkind * hostValues, int funcID);
void  validateHostDeviceResultsVolumeRenumber(int numLoops, int numElements, const fpkind * deviceValues, const fpkind * hostValues, int funcID);
void cudaMemoryFree();
#endif
