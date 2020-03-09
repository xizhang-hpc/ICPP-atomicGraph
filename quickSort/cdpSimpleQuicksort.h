__device__ void selection_sort(int *data, int * index, int left, int right);
__global__ void cdp_simple_quicksort(int *data, int * index, int left, int right, int depth);
void run_qsort(int *data, int * index, int nitems);

