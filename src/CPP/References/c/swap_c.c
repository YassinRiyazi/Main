// swap_c.c
#include <stdio.h>
#include <time.h>

#include "swap.h"

int main() {
    int x = 1, y = 2;
    clock_t start = clock();
    for (long i = 0; i < 100000000; i++) {
        swap_c(&x, &y);
    }
    clock_t end = clock();

    printf("C: x=%d, y=%d, time=%f sec\n", x, y, (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}

// gcc -O2 src/CPP/References/c/swap_c.c -o src/CPP/References/c/swap_c
// C: x=1, y=2, time=0.000233 sec



// gcc -O0 -g src/CPP/References/c/swap_c.c -o src/CPP/References/c/swap_c_dbg


//gcc -S src/CPP/References/c/swap_c.c -o src/CPP/References/c/swap_c.s