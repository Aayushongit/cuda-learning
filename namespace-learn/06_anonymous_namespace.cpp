// Example 6: Anonymous (Unnamed) Namespaces
// Used for file-private functions and variables (internal linkage)

#include <iostream>

// Anonymous namespace - contents are only visible in this file
namespace {
    int secretValue = 42;

    void helperFunction() {
        std::cout << "This is a private helper function" << std::endl;
    }

    // This constant is only accessible in this file
    const double PI = 3.14159;
}

// Regular namespace
namespace Public {
    void publicFunction() {
        std::cout << "This is a public function" << std::endl;

        // Can access anonymous namespace members in the same file
        std::cout << "Secret value: " << secretValue << std::endl;
        helperFunction();
    }

    double calculateCircleArea(double radius) {
        return PI * radius * radius;  // Using PI from anonymous namespace
    }
}

int main() {
    // Can access anonymous namespace directly in the same file
    std::cout << "Secret value: " << secretValue << std::endl;
    helperFunction();

    // Access public namespace
    Public::publicFunction();
    std::cout << "Circle area: " << Public::calculateCircleArea(5.0) << std::endl;

    return 0;
}
