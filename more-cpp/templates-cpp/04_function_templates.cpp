#include <iostream>
#include <string>
#include <vector>

// Basic function template: works with any type
template<typename T>
T getMax(T a, T b) {
    return (a > b) ? a : b;
}

template<typename N>
N getMin(N x, N y){
	return (x<y)?x :y;
	
}

// Template with multiple type parameters
template<typename T1, typename T2>
void printPair(T1 first, T2 second) {
    std::cout << "(" << first << ", " << second << ")" << std::endl;
}

// Template with non-type parameter (compile-time constant)
template<typename T, int SIZE>
T getArraySum(T (&arr)[SIZE]) {
    T sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += arr[i];
    }
    return sum;
}

// Template with default type parameter
template<typename T = int>
T getSquare(T value) {
    return value * value;
}

// Template function overloading
template<typename T>
void print(T value) {
    std::cout << "Generic: " << value << std::endl;
}

// Specialized version for pointer types
template<typename T>
void print(T* ptr) {
    std::cout << "Pointer to: " << *ptr << std::endl;
}

// Variadic template (C++11): accepts any number of arguments
template<typename T>
void printAll(T value) {
    std::cout << value << std::endl;
}

template<typename T, typename... Args>
void printAll(T first, Args... rest) {
    std::cout << first << " ";
    printAll(rest...);  // Recursive call
}

//------------------------------------------------------------

using namespace std;

// BASE CASE: When only ONE argument left
template<typename G>
void printsome(G value) {
    cout << value << endl;
}

// RECURSIVE CASE: When MULTIPLE arguments
template<typename G, typename... Args>
void printsome(G first, Args... rest) {
    cout << first << endl;
    printsome(rest...);
}

//------------------------------------------------------------

int main() {
    std::cout << "=== FUNCTION TEMPLATE EXAMPLES ===" << std::endl;

    // Template type deduction
    std::cout << "Max of 10, 20: " << getMax(10, 20) << std::endl;
    std::cout << "Max of 3.14, 2.71: " << getMax(3.14, 2.71) << std::endl;
    std::cout << "Max of 'a', 'z': " << getMax('a', 'z') << std::endl;

    // Explicit template parameter
    std::cout << "Max<double>: " << getMax<double>(5, 3.14) << std::endl;

    // Multiple template parameters
    printPair(42, "Answer");
    printPair(3.14, 2.71);
    printPair(std::string("Hello"), 100);

    // Non-type template parameter (array size deduced!)
    int numbers[] = {1, 2, 3, 4, 5};
    std::cout << "Array sum: " << getArraySum(numbers) << std::endl;

    double values[] = {1.1, 2.2, 3.3};
    double ages[]={7,80,14,71};
    
    std::cout << "Double array sum: " << getArraySum(values) << std::endl;

    // Default template parameter
    std::cout << "Square: " << getSquare(5) << std::endl;
    std::cout << "Square<double>: " << getSquare<double>(3.5) << std::endl;

    // Template overloading
    int x = 100;
    print(42);
    print(&x);

    // Variadic template
    std::cout << "Print all: ";
    printAll(1, 2.5, "hello", 'A', true);

    cout<< "new results "<< endl;

    printsome(23,6.89,"Aayush","B",false);
    

    return 0;
}
