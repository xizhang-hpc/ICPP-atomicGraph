#include "Timer.h"
	#define TIMERSIZE 200
	int timerIndex;
	int arrayTop;
	int zeroID;
	int oneID;
	struct timeval One_time, Zero_time;
	int * timerID;
	int * timerCallNum;
	double * timerElapsedTime;
	cudaEvent_t start, stop;		

	void timerStart()
	{
		//size_t sizeTimer = TIMERSIZE * sizeof(char*);
		size_t sizeTimerElapsedTime = TIMERSIZE * sizeof(double);
		size_t sizeTimerCallNum = TIMERSIZE * sizeof(int);
		timerElapsedTime = (double *)malloc(sizeTimerElapsedTime);
		timerCallNum = (int *)malloc(sizeTimerCallNum);
		//timerName = (char **)malloc(sizeTimer);
		timerID = (int *)malloc(sizeTimerCallNum);
		timerIndex = -1;	
		arrayTop = -1;
			//for cuda event method
		cudaEventCreate(&start);
                cudaEventCreate(&stop);		

	}

	void timerEnd()
	{
		for (int i = 0; i <= arrayTop; i++)
		{
			printf("Function No: %d, Timer No: %d, Time: %e s, Loop No: %d\n", 
					i, timerID[i], timerElapsedTime[i], timerCallNum[i]);	
		}
		free(timerElapsedTime);
		free(timerCallNum);
		free(timerID);
	        cudaEventDestroy(start);
        	cudaEventDestroy(stop);
	}

	void timeZero(int value)
	{
		zeroID = value;	
		int indexFlag = 0;
		for (int i = 0; i <= arrayTop; i++)
		{
			if (timerID[i] == zeroID)
			{
				timerIndex = i;
				indexFlag = 1;
				break;
			}
		}		

		if (!indexFlag) 
		{
			arrayTop++;
			timerID[arrayTop] = zeroID;
			timerElapsedTime[arrayTop] = 0.0;
			timerCallNum[arrayTop] = 0;
			timerIndex = arrayTop;
		}
		cudaDeviceSynchronize();
		gettimeofday(&Zero_time, NULL);
	}

	void timeOne(int value)
	{
		cudaDeviceSynchronize();
		gettimeofday(&One_time, NULL); 
		
		oneID = value;
		
		double elapsedTime = (One_time.tv_sec - Zero_time.tv_sec) + (One_time.tv_usec - Zero_time.tv_usec) * 1.0e-6;
		
		if (oneID != zeroID) 
		{
			printf("Error in timer: oneID is not equal to zeroID, oneID = %d, zeroID = %d\n", oneID, zeroID);
			exit(1);
		}

		timerElapsedTime[timerIndex] += elapsedTime;
		timerCallNum[timerIndex]++;
		


	}
	void timerGPUStart()
	{

		size_t sizeTimerElapsedTime = TIMERSIZE * sizeof(double);
		size_t sizeTimerCallNum = TIMERSIZE * sizeof(int);
		timerElapsedTime = (double *)malloc(sizeTimerElapsedTime);
		timerCallNum = (int *)malloc(sizeTimerCallNum);
		timerID = (int *)malloc(sizeTimerCallNum);
		timerIndex = -1;	
		arrayTop = -1;
		//for cuda event method
		cudaEventCreate(&start);
                cudaEventCreate(&stop);		
	}
	void timeGPUZero(int value)
	{

		zeroID = value;	
		int indexFlag = 0;
		for (int i = 0; i <= arrayTop; i++)
		{
			if (timerID[i] == zeroID)
			{
				timerIndex = i;
				indexFlag = 1;
				break;
			}
		}		

		if (!indexFlag) 
		{
			arrayTop++;
			timerID[arrayTop] = zeroID;
			timerElapsedTime[arrayTop] = 0.0;
			timerCallNum[arrayTop] = 0;
			timerIndex = arrayTop;
		}
		//The cuda event is only recorded for stream 0.
		cudaEventRecord(start, 0);
	}


	void timeGPUOne(int value)
	{
		cudaEventRecord(stop, 0);
                cudaEventSynchronize( stop );
		oneID = value;
		if (oneID != zeroID) 
		{
			printf("Error in timer: oneID is not equal to zeroID, oneID = %d, zeroID = %d", oneID, zeroID);
			exit(1);
		}
                float ELAPSEDTIME;
                cudaEventElapsedTime(&ELAPSEDTIME, start, stop);
		ELAPSEDTIME /=1000.0;   // from ms to s
		timerElapsedTime[timerIndex] += ELAPSEDTIME;
		timerCallNum[timerIndex]++;

	}

	void timerGPUEnd()
	{
		FILE * outputFile = fopen("timeRecord", "w");
		for (int i = 0; i <= arrayTop; i++)
		{
			//printf("Function No: %d, Algorithm: %d, Time: %e s, Loop No: %d\n", 
			//		i, timerID[i], timerElapsedTime[i], timerCallNum[i]);	
			printf("Function No: %d, Time: %e s, Loop No: %d\n", 
					i, timerElapsedTime[i], timerCallNum[i]);	
			//fprintf(outputFile, "%d  %f\n", timerID[i], timerElapsedTime[i]);
			fprintf(outputFile, "Function No. %d Performance: %f\n", i, timerElapsedTime[i]);
		}
		free(timerElapsedTime);
		free(timerCallNum);
		free(timerID);
		//delete the cuda event
	        cudaEventDestroy(start);
        	cudaEventDestroy(stop);
		fclose(outputFile);
	}
