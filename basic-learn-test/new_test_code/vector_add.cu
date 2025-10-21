#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include < cuda.h>
#include <cuda_runtime.h>

#define n 10000012
#define max_err 1e-8


__global__ void vector_add(float*out, float*a, float *b,float*c , int n ){
	int tid =blockIdx.x *blockDim.x +threadIdx.x;

	if(tid<n){
		out[tid]=a[tid]+b[tid]+c[tid];
	}
}

int main(){
	float *a, *b, *c, *out;
	float *d_a, *d_b, *d_c, *d_out;

	a=(float*)malloc(sizeof(float)*n);
	b=(float*)malloc(sizeof(float)*n);
	c=(float*)malloc(sizeof(float)*n);
	out=(float*)malloc(sizeof(float)*n);


	// now i am going to initialilze the host array

	for(int z=0;z <n;z++){
		a[z]=1.0f;
		b[z]=2.0f;
		c[z]=3.0f;
	}

	//allocating the device memory 

	cudamalloc((void**)&d_a, sizeof(float)*n);
        cudamalloc((void**)&d_b, sizeof(float)*n);
        cudamalloc((void**)&d_c, sizeof(float)*n);
        cudamalloc((void**)&d_out, sizeof(float)*n);

	cudaMemcpy(d_a,a,sizeof(float)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,sizeof(float)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_c,c,sizeof(float)*n,cudaMemcpyHostToDevice);

	int block_size= 256;
	int grid_size=((N+block_size)/block_size);

	vector_add<<<grid_size, block_size>>>(d_out,d_a,d_b,d_c,N);

	cudaMemcpy(out, d_out,sizeof(float)*n,cudaMemcpyDeviceToHost);        
	for(int p=0 , p<n; p++){
		assert(fabs(out[p]-a[p]-b[p]-c[p]) < max_err);
	}

	printf("PASSED\n");

	cudaFree(d_a);
        cudaFree(d_b);
  	cudaFree(d_c);
	cudaFree(d_out);

	free(a);
	free(b);
	free(c);
	free(out);

}







