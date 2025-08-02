// swap_cpp_call_c.cpp
#include <iostream>
#include <ctime>

extern "C" {
    void swap_c(int *a, int *b);  // Declaration of the C function
}

int main() {
    int x = 1, y = 2;
    clock_t start = clock();
    for (long i = 0; i < 1000000; i++) {
        swap_c(&x, &y);
    }
    clock_t end = clock();

    std::cout << "C++ calling C: x=" << x << ", y=" << y
              << ", time=" << (double)(end - start) / CLOCKS_PER_SEC << " sec\n";
    return 0;}
// g++ -O2 src/CPP/References/swap_cpp_call_c.cpp -o src/CPP/References/swap_cpp_call_c -L. src/CPP/References/c/swap_c.o


// gcc -c src/CPP/References/c/swap.h -O2 -o src/CPP/References/c/swap_c.o   # compile C function to object file
// g++ -O2 src/CPP/References/swap_cpp_call_c.cpp src/CPP/References/c/swap_c.o -o src/CPP/References/swap_cpp_call_c

/*
    gcc -c src/CPP/References/c/swap_c.c -O2      # compile C function to object file
    g++ -O2 src/CPP/References/swap_cpp_call_c.cpp src/CPP/References/c/swap_c.o -o src/CPP/References/swap_cpp_call_c
*/
