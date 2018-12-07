#include "commons.h"
#include "average_gpu.h"
#include "device_launch_parameters.h"


#define BLOCK_SIZE 16
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

// STD includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>


static // Print device properties
void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %zu\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %zu\n", devProp.totalConstMem);
	printf("Texture alignment:             %zu\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

void print_gpuInfo() {
	int rtVersion = 0;
	printf("*********************************************************************************************\n");
	checkCudaErrors(cudaRuntimeGetVersion(&rtVersion));
	printf("CUDA Runtime Version = %d\n", rtVersion);
	int driverVersion = 0;
	checkCudaErrors(cudaDriverGetVersion(&driverVersion));
	printf("CUDA Driver Version  = %d\n", rtVersion);

	int numDevices = 0;
	checkCudaErrors(cudaGetDeviceCount(&numDevices));
	printf("Devices found        = %d\n", numDevices);

	for (int i = 0; i < numDevices; i++) {
		cudaDeviceProp properties;
		checkCudaErrors(cudaGetDeviceProperties(&properties, i));
		printDevProp(properties);
	}
	printf("*********************************************************************************************\n");
}



/******************************************************************************
* UTILITY FUNCTIONS
******************************************************************************/


__global__ void readChannelKernel(unsigned char * image,
	unsigned char *channel,
	int imageW,
	int imageH,
	int channelToExtract,
	int numChannels) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int posIn = y  * (imageW*numChannels) + (x*numChannels) + channelToExtract;
	int posOut = y * imageW + x;

	channel[posOut] = image[posIn];


}

__global__ void writeChannelKernel(
	unsigned char* image,
	unsigned char* channel,
	int imageW,
	int imageH,
	int channelToMerge,
	int numChannels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int posOut = y * (imageW*numChannels) + (x*numChannels) + channelToMerge;
	int posIn = y * imageW + x;

	image[posOut] = channel[posIn];

}

__global__ void averageKernel(
	unsigned char* inputChannel,
	unsigned char* outputChannel,
	int imageW,
	int imageH)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numElements = ((2 * KERNEL_RADIUS) + 1) * ((2 * KERNEL_RADIUS) + 1);

	unsigned int sum = 0;
	for (int kY = -KERNEL_RADIUS; kY <= KERNEL_RADIUS; kY++) {
		const int curY = y + kY;
		if (curY < 0 || curY > imageH) {
			continue;
		}

		for (int kX = -KERNEL_RADIUS; kX <= KERNEL_RADIUS; kX++) {
			const int curX = x + kX;
			if (curX < 0 || curX > imageW) {
				continue;
			}

			const int curPosition = (curY * imageW + curX);
			if (curPosition >= 0 && curPosition < (imageW * imageH)) {
				sum += inputChannel[curPosition];
			}
		}
	}
	outputChannel[y * imageW + x] = (unsigned char)(sum / numElements);
}



/******************************************************************************
* AVERAGE FILTER
******************************************************************************/

void average_gpu(
	unsigned char* inputImage,
	unsigned char* outputImage,
	int imageW,
	int imageH,
	int numChannels) {


	unsigned char* device_input = NULL;
	unsigned char* device_output = NULL; 
	unsigned char* device_input_channel = NULL;
	unsigned char* device_output_channel = NULL;

	dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 dimGrid(imageW / dimBlock.x, imageH / dimBlock.y);
	size_t size = numChannels * imageW * imageH * sizeof(unsigned char);
	size_t sizeChannel = imageW * imageH * sizeof(unsigned char);
	checkCudaErrors(cudaMalloc((void **)&device_input, size));
	checkCudaErrors(cudaMalloc((void **)&device_output, size));
	checkCudaErrors(cudaMemcpy(device_input, inputImage, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&device_input_channel, sizeChannel));
	checkCudaErrors(cudaMalloc((void **)&device_output_channel, sizeChannel));
	int curChannel;
	cudaError_t lastError;

	printf(" average_gpu:\n");
	for (curChannel = 0; curChannel < numChannels; curChannel++) {
		printf("\tchannel(%d) readChannel", curChannel);
		readChannelKernel << <dimGrid, dimBlock >> > (device_input, device_input_channel, imageW, imageH, curChannel, numChannels);
		lastError = cudaGetLastError();
		checkCudaErrors(lastError);
		printf(" compute_average()");
		averageKernel << <dimGrid, dimBlock >> > (device_input_channel, device_output_channel, imageW, imageH);
		lastError = cudaGetLastError();
		checkCudaErrors(lastError);
		printf(" writeChannel()\n");
		writeChannelKernel << <dimGrid, dimBlock >> > (device_output, device_output_channel, imageW, imageH, curChannel, numChannels);
		lastError = cudaGetLastError();
		checkCudaErrors(lastError);
	}

	checkCudaErrors(cudaMemcpy(outputImage, device_output, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(device_input_channel));
	checkCudaErrors(cudaFree(device_output_channel));
	checkCudaErrors(cudaFree(device_input));
	checkCudaErrors(cudaFree(device_output));
}


