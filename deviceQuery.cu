#include "deviceQuery.h"
#include "cudaError.h"
void deviceQuery(){
	//cudaDeviceProp deviceProp;
	int deviceCount;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
	printf("device number: %d\n", deviceCount);

}
