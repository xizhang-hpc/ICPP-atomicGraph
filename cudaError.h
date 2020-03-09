//#include "cuda.h"
static void handleCudaError(cudaError_t error, const char * file, int line){
	if (error != cudaSuccess){
		printf("CUDA Error: %s at %s in line %d\n", cudaGetErrorString(error), file, line);
		exit(EXIT_FAILURE);
	}
}
static void handleKernelError(const char * file, int line){
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("KERNEL Error: %s at %s in line %d\n", cudaGetErrorString(error), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_CUDA_ERROR(error) (handleCudaError(error, __FILE__, __LINE__))
#define HANDLE_KERNEL_ERROR() (handleKernelError(__FILE__, __LINE__))
