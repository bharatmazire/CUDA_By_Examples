#include <stdio.h>
#include<math.h>
#include"book.h"

unsigned char* dev_bitmap;

__global__ void kernel(unsigned char* ptr, int DIM, int ticks)
{
	// map from threadIdx/BlockIdx to pixel position
	int x =  threadIdx.x + blockIdx.x * blockDim.x;
	int y =  threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	float fx 	= x - DIM/2;
	float fy 	= y - DIM/2;
	float d 	= sqrtf( fx * fx + fy * fy );

	unsigned char grey =  (unsigned char)(128.0f + 127.0f *cos(d/10.0f - ticks/7.0f) /(d/10.0f + 1.0f)); 

	ptr[offset * 4 + 0] = (unsigned char)0;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

void InitializeGPU(int DIM, unsigned char* CheckImage)
{
	static int tick = 0;
	tick += 1;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, DIM * DIM * 4));

	dim3 blocks(DIM/16, DIM/16);
	dim3 threads(16,16);

	kernel <<<blocks, threads>>> (dev_bitmap, DIM, tick);

	HANDLE_ERROR( cudaMemcpy(CheckImage, dev_bitmap, (DIM * DIM * 4), cudaMemcpyDeviceToHost) );

	cudaFree(dev_bitmap);
}
