#include<iostream>
#include<cstdlib>

using namespace std;

#define  SHMEM_SIZE(16*16)

__global__ void matrixmul(int*a, int *b, int *c, int n){

	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];
	
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	
	// Move the tile across the length of the grid!

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int dim=blockDim.x;



}


void init_matrixmul(int *m , int n ){
	for(int i=0; i<n*n; i++){
		m[i]=rand()%100;
	}

}

int main(){

	int n =1<<10;
	size_t bytes=n*n*sizeof(int);

	//allocate memory

	int *a,*b,*c;

	cudaMallocManaged(&a,bytes);
        cudaMallocManaged(&b,bytes);
   	cudaMallocManaged(&c,bytes);	

	init_matrix(a,n);
	init_matrix(b,n);

	int threads=16;
	int blocks=(n+threads-1)/threads;

	//int blocks
	dim3 THREADS(threads,threads);
	dim3 BLOCKS(blocks,blocks);
	
	matrixmul<<<BLOCKS,THREADS>>>(a,b,c,n);
	cudaDeviceSynchronize();
	
}

