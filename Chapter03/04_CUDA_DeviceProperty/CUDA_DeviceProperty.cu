/*
	CUDA program.
	CUDA_DeviceProperty.cu
	Date : 27/08/2020
	NOTE : Following code take from CUDA by Examples.
*/

// header
#include<iostream>
#include"../../Include/book.h"

int main(void)
{
	// declaration of variables
	cudaDeviceProp prop;
	int count;

	// code
	HANDLE_ERROR( cudaGetDeviceCount( &count ) );	// cudaGetDeviceCount(), to get CUDA capable device count
	printf("\n\tTotal Number of CUDA Capable Devices connected are : %d\n\n", count);

	for (int i=0; i< count; i++) 
	{
		HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );

		printf( " --- General Information for device %d ---\n", i+1 );
		printf( "Name				: %s\n", prop.name );
		printf( "Compute capability		: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate			: %d\n", prop.clockRate );
		printf( "Device copy overlap		: " );
		if (prop.deviceOverlap)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
		printf( "Kernel execition timeout	: " );
		if (prop.kernelExecTimeoutEnabled)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
		
		printf("\n--------------------------------------------------------------------------\n");
		printf( " --- Memory Information for device %d ---\n", i+1 );
		printf( "Total global mem			: %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem			: %ld\n", prop.totalConstMem );
		printf( "Max mem pitch				: %ld\n", prop.memPitch );
		printf( "Texture Alignment			: %ld\n", prop.textureAlignment );
		
		printf("\n--------------------------------------------------------------------------\n");
		printf( " --- MP Information for device %d ---\n", i+1 );
		printf( "Multiprocessor count		: %d\n",prop.multiProcessorCount );
		printf( "Shared mem per mp		: %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp		: %d\n", prop.regsPerBlock );
		printf( "Threads in warp			: %d\n", prop.warpSize );
		printf( "Max threads per block		: %d\n",prop.maxThreadsPerBlock );
		printf( "Max thread dimensions		: (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2] );
		printf( "Max grid dimensions		: (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1],prop.maxGridSize[2] );
		
		printf("\n--------------------------------------------------------------------------\n");
		printf( "\n" );
 	}
	return(0);
}
