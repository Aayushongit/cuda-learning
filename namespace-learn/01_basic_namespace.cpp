// Example 1: Basic Namespace Usage
// This shows how to create and use a simple namespace

#include <iostream>

// Define a namespace called 'Math'
namespace Math {
    int add(int a, int b) {
        return a + b;
    }

    int multiply(int a, int b) {
        return a * b;
    }
}

// Another namespace called 'Display'
namespace Display {
    void showResult(int result) {
        std::cout << "Result: " << result << std::endl;
    }
}

int main() {
    // Access namespace members using :: operator
    int sum = Math::add(5, 3);
    int product = Math::multiply(4, 6);

    Display::showResult(sum);
    Display::showResult(product);

    return 0;
}
