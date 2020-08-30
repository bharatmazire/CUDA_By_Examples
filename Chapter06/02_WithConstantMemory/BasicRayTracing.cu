#include <stdio.h>
#include<math.h>
#include"book.h"
#include"cuda.h"

#define INF 2e10f
#define rnd( x ) (x * rand() / RAND_MAX)
#define SPHERES 20

struct Sphere 
{
	float r,b,g;
	float radius;
	float x,y,z;

	__device__ float hit( float ox, float oy, float *n ) 
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) 
		{
			float dz = sqrtf( radius*radius - dx*dx - dy*dy );
			*n = dz / sqrtf( radius * radius );
			return dz + z;
		}
		return -INF;
	}
};


__constant__ Sphere s[SPHERES];

unsigned char* dev_bitmap;

__global__ void kernel(unsigned char* ptr, int DIM)
{
	// map from threadIdx/BlockIdx to pixel position
	int x =  threadIdx.x + blockIdx.x * blockDim.x;
	int y =  threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);
	
	float r=0, g=0, b=0;
	float maxz = -INF;

	for(int i = 0 ; i < SPHERES ; i++) 
	{
		float n;
		float t = s[i].hit( ox, oy, &n );
		if (t > maxz) 
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
		}
	}

	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

void InitializeGPU(int DIM, unsigned char* CheckImage)
{
	cudaEvent_t start, stop;

	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );

	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, DIM * DIM * 4));				// allocate memory for bitmap
	HANDLE_ERROR( cudaMalloc( (void**)&s,sizeof(Sphere) * SPHERES ) );			// allocate memory for spheres

	Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );

	for (int i=0; i<SPHERES; i++) 
	{
		temp_s[i].r 		= rnd( 1.0f );
		temp_s[i].g 		= rnd( 1.0f );
		temp_s[i].b 		= rnd( 1.0f );
		temp_s[i].x 		= rnd( 1000.0f ) - 500;
		temp_s[i].y 		= rnd( 1000.0f ) - 500;
		temp_s[i].z 		= rnd( 1000.0f ) - 500;
		temp_s[i].radius 	= rnd( 100.0f ) + 20;
 	}

	//HANDLE_ERROR( cudaMemcpy( s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpyToSymbol( s, temp_s, sizeof(Sphere) * SPHERES) );							// for constant memory
	
	free( temp_s );

	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16,16);

	kernel <<<grids, threads>>> (dev_bitmap, DIM);

	HANDLE_ERROR( cudaMemcpy(CheckImage, dev_bitmap, (DIM * DIM * 4), cudaMemcpyDeviceToHost) );

	cudaFree(dev_bitmap);
	cudaFree(s);
}
