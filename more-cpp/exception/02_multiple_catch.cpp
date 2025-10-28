/**
 * 02_multiple_catch.cpp
 *
 * TOPIC: Multiple Catch Blocks
 *
 * This file demonstrates:
 * - Handling different types of exceptions
 * - Multiple catch blocks
 * - Catch-all handler (...)
 * - Order of catch blocks matters
 */

#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

// Function that throws different types of exceptions
void processInput(int choice) {
    switch (choice) {
        case 1:
            throw runtime_error("Runtime error occurred!");
        case 2:
            throw invalid_argument("Invalid argument provided!");
        case 3:
            throw out_of_range("Index out of range!");
        case 4:
            throw 42;  // Throwing an integer
        case 5:
            throw string("String exception!");
        case 6:
            throw 3.14;  // Throwing a double
        default:
            cout << "No exception thrown" << endl;
    }
}

int main() {
    cout << "=== Multiple Catch Blocks ===" << endl;

    // Example 1: Catching specific exception types
    cout << "\nExample 1: Different Exception Types" << endl;

    for (int choice = 1; choice <= 7; choice++) {
        cout << "\n--- Testing choice " << choice << " ---" << endl;

        try {
            processInput(choice);
            cout << "No exception was thrown" << endl;
        }
        catch (const runtime_error& e) {
            cout << "Caught runtime_error: " << e.what() << endl;
        }
        catch (const invalid_argument& e) {
            cout << "Caught invalid_argument: " << e.what() << endl;
        }
        catch (const out_of_range& e) {
            cout << "Caught out_of_range: " << e.what() << endl;
        }
        catch (int value) {
            cout << "Caught integer exception: " << value << endl;
        }
        catch (const string& str) {
            cout << "Caught string exception: " << str << endl;
        }
        catch (...) {
            // Catch-all handler: catches ANY exception type
            cout << "Caught unknown exception type (catch-all handler)" << endl;
        }
    }

    // Example 2: Order of catch blocks matters
    cout << "\n\nExample 2: Exception Hierarchy" << endl;
    cout << "Note: More specific exceptions should be caught before general ones" << endl;

    try {
        throw out_of_range("Array index out of bounds!");
    }
    catch (const out_of_range& e) {
        // More specific exception - caught first
        cout << "Specific: " << e.what() << endl;
    }
    catch (const logic_error& e) {
        // More general exception - would catch out_of_range too
        cout << "General: " << e.what() << endl;
    }
    catch (const exception& e) {
        // Most general - would catch all standard exceptions
        cout << "Most general: " << e.what() << endl;
    }

    // Example 3: Using catch-all as safety net
    cout << "\n\nExample 3: Catch-All as Safety Net" << endl;
    try {
        throw "Unexpected exception!";  // C-string
    }
    catch (const exception& e) {
        cout << "Standard exception: " << e.what() << endl;
    }
    catch (...) {
        cout << "Caught non-standard exception - safety net activated!" << endl;
    }

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. Multiple catch blocks can handle different exception types
     * 2. First matching catch block is executed
     * 3. Catch blocks are checked in order from top to bottom
     * 4. catch(...) catches ALL exception types
     * 5. Order matters: specific exceptions before general ones
     * 6. Only ONE catch block is executed per exception
     * 7. You can throw any type: objects, primitives, pointers
     */

    return 0;
}
