#include "stdio.h"
#include "precision.h"
void CallGraphColoringMax(int numVolumes);
__global__ void kernelInitGraphColoring(int numVolumes, fpkind  * pressureLocalMax);
__global__ void kernelGraphColoringMax(int numFaces, int posiGroup, const int * faceGroup, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax);
__device__ fpkind kernelMAXDOUBLEGraph(fpkind a, fpkind b);
__global__ void kernelGraphColoringMaxOpt(int numFaces, int posiGroup, const int * faceGroup, const int * owner, const int * neighbour, const fpkind * pressure, fpkind * pressureLocalMax);
