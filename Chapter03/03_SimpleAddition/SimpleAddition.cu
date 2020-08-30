/*
	CUDA program.
	SimpleAddition.cu
	Date : 25/08/2020
*/

// header
#include<iostream>
#include"../../Include/book.h"


__global__ void add(int a, int b, int *c)
{
	// code
	*c = a + b;
}

int main(void)
{
	// declaration of variables
	int c;
	int *dev_c;

	// code
	// HANDLE_ERROR : macro from book.h, for error check
	HANDLE_ERROR( cudaMalloc((void**)&dev_c, sizeof(int)) );	// cudaMalloc() to allocate the memory on GPU / Device
	// The first argument is a pointer to the pointer you want to hold the address of the newly allocated memory, 
	// and the second parameter is the size of the allocation you want to make
	


	add<<<1,1>>>(2,7,dev_c);			// call kernel with function parameters in '(...)'

	HANDLE_ERROR( cudaMemcpy(&c,dev_c, sizeof(int), cudaMemcpyDeviceToHost) );		// cudaMemcpy() to copy memory from(and to) GPU. cudaMemcpyDeviceToHost : Device to Host => GPU to CPU

	printf("\n\t 2 + 7 is : %d \n\n",c);

	cudaFree(dev_c);				// free the cuda memory
	
	return(0);
}
