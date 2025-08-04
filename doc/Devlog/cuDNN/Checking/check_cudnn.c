#include <cudnn.h>
#include <stdio.h>
int main() {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status == CUDNN_STATUS_SUCCESS) {
        printf("cuDNN is installed and working!\n");
        cudnnDestroy(handle);
    } else {
        printf("cuDNN error: %s\n", cudnnGetErrorString(status));
    }
    return 0;
}

/*
gcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o check_cudnn check_cudnn.c -lcudnn 
./check_cudnn
*/
