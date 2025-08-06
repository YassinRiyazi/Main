

#include <iostream>
union Converter {
    int i;
    float f;
    char c[4];  // Ensure enough space for both int and float
};

int main() {
    /*
    Application of union in C++.
    A union allows storing different data types in the same memory location.
    The size of the union is determined by the largest member.
    Here, we define a union named Converter that can hold either an int or a float.
    The union's size is the size of the largest member, which is float in this case.
    Accessing the members of a union is done through the same memory location,
    The last written member determines the value of the union.
    In this example, we write a float value and then read it as an int.
    This is a demonstration of how unions can be used to interpret the same memory in different ways.
    Note: Using unions requires caution as it can lead to undefined behavior if not used correctly.
    The union is a powerful feature in C++ that allows for memory-efficient data representation.
    */
    Converter c;
    c.f = 3.14f;
    std::cout << c.i << "\n";  // View float bits as int
    std::cout << c.c << "\n";  // View float bits as char array
    return 0;
}

/* g++ src/CPP/TourOfCPP/C2/Union.cpp -o  src/CPP/TourOfCPP/C2/Union */