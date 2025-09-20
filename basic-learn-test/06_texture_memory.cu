#include <stdio.h>
#include <cuda_runtime.h>

texture<float, 2, cudaReadModeElementType> tex;

__global__ void bilinear_interpolation(float *output, int width, int height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float u = x * scale;
        float v = y * scale;
        
        float result = tex2D(tex, u + 0.5f, v + 0.5f);
        output[y * width + x] = result;
    }
}

int main() {
    const int orig_width = 16;
    const int orig_height = 16;
    const int new_width = 32;
    const int new_height = 32;
    const float scale = (float)orig_width / new_width;
    
    float *h_input = (float*)malloc(orig_width * orig_height * sizeof(float));
    float *h_output = (float*)malloc(new_width * new_height * sizeof(float));
    
    for (int i = 0; i < orig_height; i++) {
        for (int j = 0; j < orig_width; j++) {
            h_input[i * orig_width + j] = i * orig_width + j;
        }
    }
    
    cudaArray *d_input;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&d_input, &channelDesc, orig_width, orig_height);
    cudaMemcpyToArray(d_input, 0, 0, h_input, orig_width * orig_height * sizeof(float), cudaMemcpyHostToDevice);
    
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;
    
    cudaBindTextureToArray(tex, d_input, channelDesc);
    
    float *d_output;
    cudaMalloc(&d_output, new_width * new_height * sizeof(float));
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((new_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (new_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    bilinear_interpolation<<<blocksPerGrid, threadsPerBlock>>>(d_output, new_width, new_height, scale);
    
    cudaMemcpy(h_output, d_output, new_width * new_height * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Texture memory bilinear interpolation:\n");
    printf("Original 4x4 corner:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.0f ", h_input[i * orig_width + j]);
        }
        printf("\n");
    }
    
    printf("Interpolated 4x4 corner:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.1f ", h_output[i * new_width + j]);
        }
        printf("\n");
    }
    
    cudaUnbindTexture(tex);
    cudaFreeArray(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}