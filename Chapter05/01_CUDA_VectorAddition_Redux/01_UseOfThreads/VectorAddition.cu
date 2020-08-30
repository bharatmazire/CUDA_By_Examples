/*
	CUDA program.
	VectorAddition.cu
	Date : 28/08/2020
*/

// header
#include<iostream>
#include"../../../Include/book.h"

#define N 10

__global__ void add(int *a, int *b, int *c)
{
	// declaration of variables
	int tid;		// for thread id

	// code
	tid = threadIdx.x;

	if(tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}

int main(void)
{
	// declaration of variables
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// code

	// allocate the memory on GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// fill the array a and b
	for(int i = 0 ; i < N ; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}
	
	// copy the arrays a and b to the GPU
	HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) );

	add<<<1,N>>>(dev_a,dev_b,dev_c);

	// copy c array from GPU to host
	HANDLE_ERROR( cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost) );

	for(int i = 0 ; i < N; i++)
	{
		printf("%d + %d = %d \n",a[i], b[i], c[i]);
	}

	// free the device memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
 	
	return(0);
}
