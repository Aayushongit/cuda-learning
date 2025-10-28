// Example 5: Using Directive and Using Declaration
// Ways to avoid typing namespace:: repeatedly

#include <iostream>

namespace Calculator {
    int add(int a, int b) { return a + b; }
    int subtract(int a, int b) { return a - b; }
    int multiply(int a, int b) { return a * b; }
}

namespace Printer {
    void print(int value) {
        std::cout << "Value: " << value << std::endl;
    }
}

// Method 1: using directive (imports entire namespace)
void method1() {
    using namespace Calculator;  // Now all Calculator members are accessible

    int result = add(10, 5);     // No need for Calculator::
    Printer::print(result);
}

// Method 2: using declaration (imports specific member)
void method2() {
    using Calculator::multiply;  // Only multiply is accessible without ::
    using Printer::print;        // Only print is accessible without ::

    int result = multiply(7, 3);
    print(result);

    // Note: add() and subtract() still need Calculator::
    Calculator::add(1, 2);
}

// Method 3: using in local scope
void method3() {
    {
        using namespace Calculator;
        int result = subtract(20, 8);
        Printer::print(result);
    }
    // Calculator namespace is no longer accessible here
}

int main() {
    method1();
    method2();
    method3();

    return 0;
}
