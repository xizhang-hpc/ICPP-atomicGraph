#include "stdio.h"
#include "precision.h"
void CallGraphColoringAdd(int numVolumes);
__global__ void kernelGraphColoringAdd(int numFaces, int posiGroup, const int * faceGroup, 
		const int * owner, const int * neighbour, const fpkind * flux, fpkind * residual);
__global__ void kernelGraphColoringAddFluxOpt(int numFaces, int posiGroup, const int * faceGroup, 
		const int * owner, const int * neighbour, const fpkind * flux, fpkind * residual);
__global__ void kernelGraphColoringAddShareOpt(int numFaces, int posiGroup, const int * faceGroup, 
		const int * owner, const int * neighbour, const fpkind * flux, fpkind * residual);

