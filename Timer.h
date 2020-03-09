#pragma once
#include "stdio.h"
#include <ctime>
#include <cuda_runtime.h>
#ifndef WIN32
#include <sys/time.h>
#endif

	extern int zeroID;
	extern int oneID;
	extern int timerIndex;
	extern int arrayTop;
	extern struct timeval One_time, Zero_time;
	extern int * timerID;
	extern int * timerCallNum;
	extern double * timerElapsedTime;
	extern cudaEvent_t start, stop;		
	void timerStart();
	void timeZero(int);	
	void timeOne(int);
	void timerEnd();
	void timerGPUStart();
	void timeGPUZero(int);
	void timeGPUOne(int);
	void timerGPUEnd();
