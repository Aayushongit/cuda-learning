#include <iostream>
#include <string>
#include <cstring>

// Primary template: works for most types
template<typename T>
class Printer {
public:
    void print(T value) {
        std::cout << "Generic: " << value << std::endl;
    }
};

// Full specialization for char*
template<>
class Printer<char*> {
public:
    void print(char* value) {
        std::cout << "C-String: " << value << " (length: " << strlen(value) << ")" << std::endl;
    }
};

// Full specialization for bool
template<>
class Printer<bool> {
public:
    void print(bool value) {
        std::cout << "Boolean: " << (value ? "TRUE" : "FALSE") << std::endl;
    }
};

// Primary function template
template<typename T>
T getAbsolute(T value) {
    return (value < 0) ? -value : value;
}

// Specialization for std::string (abs doesn't make sense, return length)
template<>
std::string getAbsolute<std::string>(std::string value) {
    return "String length: " + std::to_string(value.length());
}

// Generic comparison
template<typename T>
bool isEqual(T a, T b) {
    return a == b;
}

// Specialization for floating point (use epsilon comparison)
template<>
bool isEqual<double>(double a, double b) {
    const double epsilon = 0.00001;
    return std::abs(a - b) < epsilon;
}

// Partial specialization (only works with classes, not functions)
template<typename T, typename U>
class Container {
public:
    void identify() {
        std::cout << "Container with two different types" << std::endl;
    }
};

// Partial specialization when both types are the same
template<typename T>
class Container<T, T> {
public:
    void identify() {
        std::cout << "Container with two SAME types" << std::endl;
    }
};

// Partial specialization for pointer types
template<typename T>
class Container<T*, T*> {
public:
    void identify() {
        std::cout << "Container with two pointer types" << std::endl;
    }
};

// SFINAE: Substitution Failure Is Not An Error
// Enable function only for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
printType(T value) {
    std::cout << value << " is an integral type" << std::endl;
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
printType(T value) {
    std::cout << value << " is a floating point type" << std::endl;
}

int main() {
    std::cout << "=== TEMPLATE SPECIALIZATION EXAMPLES ===" << std::endl;

    // Class template specialization
    Printer<int> intPrinter;
    intPrinter.print(42);

    Printer<double> doublePrinter;
    doublePrinter.print(3.14159);

    char str[] = "Hello";
    Printer<char*> strPrinter;
    strPrinter.print(str);

    Printer<bool> boolPrinter;
    boolPrinter.print(true);
    boolPrinter.print(false);

    std::cout << std::endl;

    // Function template specialization
    std::cout << "Abs(-5): " << getAbsolute(-5) << std::endl;
    std::cout << "Abs(-3.14): " << getAbsolute(-3.14) << std::endl;
    std::cout << getAbsolute<std::string>("Hello World") << std::endl;

    std::cout << std::endl;

    // Specialized comparison
    std::cout << "5 == 5: " << isEqual(5, 5) << std::endl;
    std::cout << "3.14159 == 3.14160 (epsilon): " << isEqual(3.14159, 3.14160) << std::endl;

    std::cout << std::endl;

    // Partial specialization
    Container<int, double> c1;
    c1.identify();

    Container<int, int> c2;
    c2.identify();

    Container<int*, int*> c3;
    c3.identify();

    std::cout << std::endl;

    // SFINAE examples
    printType(42);
    printType(3.14);

    std:: cout<< "SOME DEFINED MACROS "<<std:: endl;

    std:: cout<< __FILE__<< std::endl;
    std:: cout<< __LINE__<< std::endl;
    std:: cout<< __DATE__<< std::endl;
    std:: cout<< __TIME__<< std::endl;
 //   std:: cout<< __cplusplus<< std::endl;

    return 0;
}
