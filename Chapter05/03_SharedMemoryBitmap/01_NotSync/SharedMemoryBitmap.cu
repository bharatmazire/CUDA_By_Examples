#include <stdio.h>
#include<math.h>
#include"book.h"

#define PI 3.14285714285714

unsigned char* dev_bitmap;

__global__ void kernel(unsigned char* ptr, int DIM)
{
	// map from threadIdx/BlockIdx to pixel position
	int x =  threadIdx.x + blockIdx.x * blockDim.x;
	int y =  threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	__shared__ float shared[16][16];


	// now calculate the value at that position
	const float period = 128.0f;

	shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) *(sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

	ptr[offset * 4 + 0] = (unsigned char)0;
	ptr[offset * 4 + 1] = (unsigned char)shared[15-threadIdx.x][15-threadIdx.y];
	ptr[offset * 4 + 2] = (unsigned char)0;
	ptr[offset * 4 + 3] = 255;
}

void InitializeGPU(int DIM, unsigned char* CheckImage)
{
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, DIM * DIM * 4));

	dim3 blocks(DIM/16, DIM/16);
	dim3 threads(16,16);

	kernel <<<blocks, threads>>> (dev_bitmap, DIM);

	HANDLE_ERROR( cudaMemcpy(CheckImage, dev_bitmap, (DIM * DIM * 4), cudaMemcpyDeviceToHost) );

	cudaFree(dev_bitmap);
}
