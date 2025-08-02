// swap_cpp.cpp
#include <iostream>
#include <ctime>

void swap_cpp(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 1, y = 2;
    clock_t start = clock();
    for (long i = 0; i < 1000000; i++) {
        swap_cpp(x, y);
    }
    clock_t end = clock();

    std::cout << "C++: x=" << x << ", y=" << y
              << ", time=" << (double)(end - start) / CLOCKS_PER_SEC << " sec\n";
    return 0;
}
// g++ -O2 src/CPP/Refrences/cpp/swap_cpp.cpp -o src/CPP/Refrences/cpp/swap_cpp
// C++: x=1, y=2, time=0.000348 sec