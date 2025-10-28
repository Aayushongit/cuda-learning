/**
 * 01_basic_exception.cpp
 *
 * TOPIC: Basic Exception Handling with try-catch
 *
 * This file demonstrates:
 * - What exceptions are
 * - Basic try-catch syntax
 * - When exceptions are thrown
 * - How program flow changes with exceptions
 */

#include <iostream>
#include <stdexcept>

using namespace std;

// Function that might throw an exception
double divide(double numerator, double denominator) {
    if (denominator == 0) {
        // Throw an exception when error occurs
        throw runtime_error("Division by zero is not allowed!");
    }
    return numerator / denominator;
}

int main() {
    cout << "=== Basic Exception Handling ===" << endl;

    // Example 1: Normal execution (no exception)
    cout << "\nExample 1: Normal Division" << endl;
    try {
        double result = divide(10, 2);
        cout << "Result: " << result << endl;
        cout << "This line executes normally" << endl;
    }
    catch (const runtime_error& e) {
        cout << "Error caught: " << e.what() << endl;
    }

    // Example 2: Exception is thrown
    cout << "\nExample 2: Division by Zero" << endl;
    try {
        cout << "Before division" << endl;
        double result = divide(10, 0);  // This throws an exception
        cout << "After division: " << result << endl;  // This line is NEVER executed
    }
    catch (const runtime_error& e) {
        cout << "Error caught: " << e.what() << endl;
        cout << "Program continues after handling exception" << endl;
    }

    // Example 3: Without exception handling (commented out for safety)
    cout << "\nExample 3: What happens without try-catch?" << endl;
    cout << "If we don't use try-catch, the program will terminate abruptly!" << endl;

    // Uncomment the line below to see program crash:
    // double result = divide(10, 0);  // Program crashes!

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. try block: Contains code that might throw an exception
     * 2. catch block: Handles the exception if one is thrown
     * 3. throw: Sends an exception to be caught
     * 4. When exception is thrown, execution jumps to catch block
     * 5. Code after throw in try block is NOT executed
     * 6. Program continues normally after catch block
     */

    return 0;
}
