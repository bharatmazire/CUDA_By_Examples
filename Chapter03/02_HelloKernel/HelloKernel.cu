/*
	2nd CUDA program.
	HelloKernel.cu
	Date : 25/08/2020
*/

// header
#include<iostream>
#include"../../Include/book.h"


__global__ void kernel(void)
// __global__ means : Function compiled to run on device, not on host 
{

}

int main(void)
{
	// code
	kernel<<<1,1>>>();

	printf("\n\t Hello, Kernel !!!\n\n");
	return(0);
}
