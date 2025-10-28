/**
 * 03_standard_exceptions.cpp
 *
 * TOPIC: Standard Exception Classes
 *
 * This file demonstrates:
 * - C++ Standard Library exception hierarchy
 * - Common exception types and when to use them
 * - Using std::exception as base class
 * - Practical examples of each exception type
 */

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <climits>
#include <limits>

using namespace std;

/**
 * C++ STANDARD EXCEPTION HIERARCHY:
 *
 * std::exception (base class)
 * |
 * |-- logic_error (programming errors, preventable)
 * |   |-- invalid_argument
 * |   |-- domain_error
 * |   |-- length_error
 * |   |-- out_of_range
 * |   |-- future_error
 * |
 * |-- runtime_error (runtime problems, not preventable)
 * |   |-- range_error
 * |   |-- overflow_error
 * |   |-- underflow_error
 * |   |-- system_error
 * |
 * |-- bad_alloc (memory allocation failure)
 * |-- bad_cast (dynamic_cast failure)
 * |-- bad_typeid (typeid of null pointer)
 * |-- bad_exception
 */

// Example functions demonstrating different exception types

// 1. invalid_argument: Invalid argument passed to function
double calculateSquareRoot(double value) {
    if (value < 0) {
        throw invalid_argument("Cannot calculate square root of negative number!");
    }
    return sqrt(value);
}

// 2. out_of_range: Index or key out of valid range
int getElement(const vector<int>& vec, size_t index) {
    if (index >= vec.size()) {
        throw out_of_range("Index " + to_string(index) + " is out of range!");
    }
    return vec[index];
}

// 3. length_error: Operation would exceed maximum size
void addElements(vector<int>& vec, size_t count) {
    if (count > vec.max_size()) {
        throw length_error("Cannot add " + to_string(count) + " elements - exceeds max size!");
    }
    for (size_t i = 0; i < count; i++) {
        vec.push_back(i);
    }
}

// 4. domain_error: Mathematical domain error
double calculateLogarithm(double value) {
    if (value <= 0) {
        throw domain_error("Logarithm domain error: value must be positive!");
    }
    return log(value);
}

// 5. runtime_error: General runtime error
void processFile(const string& filename) {
    if (filename.empty()) {
        throw runtime_error("Cannot process empty filename!");
    }
    // Simulate file processing
    cout << "Processing file: " << filename << endl;
}

// 6. overflow_error: Arithmetic overflow
int multiply(int a, int b) {
    if (a > 0 && b > 0 && a > INT_MAX / b) {
        throw overflow_error("Integer overflow in multiplication!");
    }
    return a * b;
}

// 7. underflow_error: Arithmetic underflow
double divideDouble(double numerator, double denominator) {
    if (abs(denominator) > 0 && abs(numerator / denominator) < numeric_limits<double>::min()) {
        throw underflow_error("Result too small to represent!");
    }
    return numerator / denominator;
}

int main() {
    cout << "=== Standard Exception Classes ===" << endl;

    // Example 1: invalid_argument
    cout << "\n1. invalid_argument:" << endl;
    try {
        double result = calculateSquareRoot(-5);
    }
    catch (const invalid_argument& e) {
        cout << "   Error: " << e.what() << endl;
    }

    // Example 2: out_of_range
    cout << "\n2. out_of_range:" << endl;
    try {
        vector<int> numbers = {10, 20, 30};
        int value = getElement(numbers, 10);
    }
    catch (const out_of_range& e) {
        cout << "   Error: " << e.what() << endl;
    }

    // Example 3: length_error
    cout << "\n3. length_error:" << endl;
    try {
        vector<int> data;
        addElements(data, data.max_size() + 1);
    }
    catch (const length_error& e) {
        cout << "   Error: " << e.what() << endl;
    }

    // Example 4: domain_error
    cout << "\n4. domain_error:" << endl;
    try {
        double result = calculateLogarithm(-10);
    }
    catch (const domain_error& e) {
        cout << "   Error: " << e.what() << endl;
    }

    // Example 5: runtime_error
    cout << "\n5. runtime_error:" << endl;
    try {
        processFile("");
    }
    catch (const runtime_error& e) {
        cout << "   Error: " << e.what() << endl;
    }

    // Example 6: overflow_error
    cout << "\n6. overflow_error:" << endl;
    try {
        int result = multiply(INT_MAX, 2);
    }
    catch (const overflow_error& e) {
        cout << "   Error: " << e.what() << endl;
    }

    // Example 7: Catching by base class
    cout << "\n7. Catching by base class (std::exception):" << endl;
    try {
        throw domain_error("Some error occurred");
    }
    catch (const exception& e) {
        // Can catch any standard exception
        cout << "   Caught via base class: " << e.what() << endl;
    }

    // Example 8: bad_alloc (memory allocation failure)
    cout << "\n8. bad_alloc:" << endl;
    try {
        // Try to allocate huge amount of memory
        size_t huge = 1ULL << 62;  // Extremely large size
        int* ptr = new int[huge];
        delete[] ptr;
    }
    catch (const bad_alloc& e) {
        cout << "   Memory allocation failed: " << e.what() << endl;
    }

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. Use logic_error for programming bugs (preventable)
     * 2. Use runtime_error for runtime conditions (not preventable)
     * 3. invalid_argument: Bad function arguments
     * 4. out_of_range: Index/key out of bounds
     * 5. domain_error: Mathematical domain violations
     * 6. length_error: Size/length constraints violated
     * 7. overflow_error/underflow_error: Arithmetic errors
     * 8. All standard exceptions inherit from std::exception
     * 9. Catch by const reference: catch(const exception& e)
     * 10. Use e.what() to get error message
     */

    return 0;
}
