#include <stdio.h>
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%d.%d\n", prop.major, prop.minor);
    return 0;
}