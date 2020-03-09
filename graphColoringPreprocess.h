#include "stdio.h"
extern int numFaceGroups;
extern int * faceGroupNums;
extern int * posiFaceGroup;
extern int * d_faceGroup;
extern int * d_faceGroupOpt; //optimize by renumber
void graphColoringPreprocess(int numVolumes, int numFaces, const int * cellFaces, const int * posiCellFaces,  int * owner, int * neighbour);
void CreateFaceConflict(int numVolumes, int numFaces, const int * cellFaces, const int * posiCellFaces, const int * owner, const int * neighbour);
void ColorFaces(int numFaces);
void hToDFaceGroupInfoTransfer(int numFaces, const int * h_faceGroup);
void hToDFaceGroupOptInfoTransfer(int numFaces, const int * h_faceGroup);
void freeHostMemory(int * faceConflict, int * posiFaceConflict, int * faceGroup);
void freeMemoryGraphColoring();
void faceGroupRenumberByOwner(int numFaceGroups, const int * faceGroupNums, int * faceGroup, const int * posiFaceGroup, const int * owner);
void cellRenumberByfaceGroup(int numFaces, int numVolumes, int numFaceGroups, const int * faceGroup, const int * posiFaceGroup, const int * faceGroupNums, int * owner, int * neighbour);
void validateFaceGroup(int numFaces, int numVolumes, int numFaceGroups, const int * faceGroup, const int * posiFaceGroup, const int * faceGroupNums, const int * owner, const int * neighbour);
void faceGroupRenumberByNeighbour(int numFaceGroups, const int * faceGroupNums, int * faceGroup, const int * posiFaceGroup, const int * neighbour);
void faceGroupRenumberQuickSortOwner(int numFaceGroups, const int * faceGroupNums, int * faceGroup, const int * posiFaceGroup, const int * owner);
void initVolumeRenumberNew(int numVolumes);
