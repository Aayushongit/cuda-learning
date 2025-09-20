#include<stdio.h>
#include<math.h>
#include<iostream>
#include<cstdlib>
#include<cassert>

using namespace std;

__global__ void matrixmul(int *a, int *b, int*c, int n ){

	//calculating the global row and column

	int row=blockIdx.y* blockDim.y+threadIdx.y;
	int col=blockIdx.x* blockDim.x+threadIdx.x;

	//boundary check matrix 

	if(row<n && col<n){
		int tmp=0;
		for(int j=0; j<n; j++){
		tmp+=a[row*n+j]*b[j*n+col];


		}
	}
}


//building matrices 

void init_matrix(int*m, int n){
	for(int i =0; i<n; i++){
		m[i]=rand()%100;
	}
}

void verify_result(int *a, int *b, int *c , int n ){
	int tmp;
	for(int i=0 ; i<n; i++){
		for(int j=0; j<n; j++){
			tmp=0;
			for(int k=0; k<n; k++){
				tmp+=a[i*n+k]*b[k*n+j];

			}
			assert(tmp==c[i*n+j]);

		}
	}


}

int main(){
	int n =1<<10;
	size_t bytes=n*n*sizeof(int);

	//allocate memory for our matrices

	int *a, *b,*c;
	cudaMallocManaged(&a,bytes);
	cudaMallocManaged(&b,bytes);
	cudaMallocManaged(&c,bytes);

	//initialize our matrices 

	init_matrix(a,n);
	init_matrix(b,n);
	
	//seting our cta and grid dim 

	int thread=16;
	int blocks=(n+thread-1)/thread;


	// setup our kernel launch parameters

	dim3 THREADS(thread,thread);
	dim3 BLOCKS(blocks,blocks);

	//launching our kernel 

	matrixmul<<<BLOCKS,THREADS>>>(a,b,c,n);

	cudaDeviceSynchronize();

	verify_result(a, b,c,n);

	cout<< "PROGRAM COMPLETED SUCCESSFULLY !"<<endl;

	return 0;


}
