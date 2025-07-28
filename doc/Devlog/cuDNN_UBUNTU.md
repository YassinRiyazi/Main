
1. !wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
2. !sudo dpkg -i cuda-keyring_1.1-1_all.deb
3. !sudo apt-get update
4. !sudo apt-get -y install cudnn


Can be checked with:
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

    !gcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o check_cudnn check_cudnn.c -lcudnn 
    !./check_cudnn