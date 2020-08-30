
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


unsigned char* dev_bitmap;

struct cuComplex
{
	float r;
	float i;

	__device__ cuComplex(float a, float b) : r(a), i(b)
	{

	}
	__device__ float magnitude2(void)
	{
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
};


__device__ int julia(int x, int y, int DIM)
{
	const float scale = 1.5;

	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__global__ void kernel(unsigned char* ptr, int DIM)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// now calculate the value at that position
	int juliaValue = julia(x, y, DIM);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

void InitializeGPU(int DIM, unsigned char* CheckImage)
{
	cudaMalloc((void**)&dev_bitmap, DIM * DIM * 4);

	dim3 grid(DIM, DIM);
	kernel <<<grid, 1>>> (dev_bitmap, DIM);

	cudaMemcpy(CheckImage, dev_bitmap, (DIM * DIM * 4), cudaMemcpyDeviceToHost);

	cudaFree(dev_bitmap);
}
