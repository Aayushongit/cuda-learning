#include <iostream>
#include <vector>
#include <map>

// typedef: Old C-style way to create type aliases
typedef unsigned long ulong;
typedef int* IntPtr;
typedef std::vector<int> IntVector;

// typedef with function pointers (syntax gets ugly)
typedef void (*FunctionPtr)(int, int);

// typedef with complex nested types
typedef std::map < std::string, std::vector<int> > StringToIntVectorMap;

typedef double (*funptr)(float, float, float);

double mulnum(float x, float y , float z ){
	return x*y*z;
	
}


void printSum(int a, int b) {
    std::cout << "Sum: " << (a + b) << std::endl;
}

int main() {
    std::cout << "=== TYPEDEF EXAMPLES ===" << std::endl;

    // Using typedef for primitive type alias
    ulong bigNumber = 9999999999UL;
    std::cout << "ulong value: " << bigNumber << std::endl;

    // Using typedef for pointer type
    int value = 42;
    IntPtr ptr = &value;
    std::cout << "IntPtr points to: " << *ptr << std::endl;

    // Using typedef for container
    IntVector numbers = {1, 2, 3, 4, 5};
    std::cout << "IntVector size: " << numbers.size() << std::endl;

    // Using typedef for function pointer
    FunctionPtr funcPtr = printSum;
    funcPtr(10, 20);

    funptr oye=mulnum;
    double sum=oye(23.57,7.096,1.0970);
    std::cout<<sum<< std:: endl;
    
    

    // Using typedef for complex nested type
    StringToIntVectorMap myMap;
    myMap["primes"] = {2, 3, 5, 7};
    myMap["evens"] = {2, 4, 6, 8};
    std::cout << "Map has " << myMap.size() << " entries" << std::endl;

    return 0;
}
