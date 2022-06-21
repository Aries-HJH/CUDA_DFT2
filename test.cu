#include <stdio.h>

__global__ void hello_from_gpu() {
	const int bidx = blockIdx.x;
	const int bidy = blockIdx.y;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	printf("Hello World from block (%d, %d) and thread (%d, %d)!\n", bidx, bidy, tidx, tidy);
}

int main(void) {
	dim3 grid_size(2, 2);
	dim3 block_size(3, 2);
	hello_from_gpu<<<grid_size, block_size>>>();
	cudaDeviceSynchronize();
	return 0;
}
