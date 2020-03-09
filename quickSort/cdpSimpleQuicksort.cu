/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include "cdpSimpleQuicksort.h"
#include <iostream>
#include <cstdio>

#define MAX_DEPTH       16
#define INSERTION_SORT  32
//#define INSERTION_SORT  100

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(int *data, int * index, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        int  min_val = data[i];
        int min_idx = i;
	//store the index of smallest value
	int indexMinValue = index[i];
        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            int val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
		indexMinValue = index[j];
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
	    //swap index
	    index[min_idx] = index[i];
	    index[i] = indexMinValue;
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(int *data, int * index, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        //selection_sort(data, left, right);
        selection_sort(data, index, left, right);
        return;
    }

    int *lptr = data+left;
    int *rptr = data+right;
    int  pivot = data[(left+right)/2];
    //for index change
    int *indexLptr = index + left;
    int *indexRptr = index + right;
    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        int lval = *lptr;
        int rval = *rptr;
        // Find the next indexLeft- and indexRight-hand values to swap
        int indexLval = *indexLptr;
        int indexRval = *indexRptr;
        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
	    //index pointer should go forward as well
	    indexLptr++;
	    indexLval = *indexLptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
	    //index pointer should go backward as well
	    indexRptr--;
	    indexRval = *indexRptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
	    *indexLptr++ = indexRval;
	    *indexRptr-- = indexLval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        //cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, index, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        //cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, index, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}
////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(int *data, int * index, int nitems)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
   
    // Launch on device
    int left = 0;
    int right = nitems-1;
    //cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
    cdp_simple_quicksort<<< 1, 1 >>>(data, index, left, right, 0);
    cudaDeviceSynchronize();
}

