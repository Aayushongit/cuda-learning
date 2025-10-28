/**
 * 05_rethrow_exceptions.cpp
 *
 * TOPIC: Rethrowing Exceptions
 *
 * This file demonstrates:
 * - Rethrowing exceptions with throw;
 * - Partial exception handling
 * - Exception chaining/wrapping
 * - Preserving exception information
 * - When and why to rethrow
 */

#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>

using namespace std;

// Custom exception that can wrap another exception
class WrappedException : public exception {
private:
    string message;
    string originalMessage;

public:
    WrappedException(const string& msg, const string& original)
        : message(msg), originalMessage(original) {}

    const char* what() const noexcept override {
        return message.c_str();
    }

    string getOriginalMessage() const { return originalMessage; }
};

// Example 1: Simple rethrow
void lowLevelOperation() {
    cout << "   lowLevelOperation: Throwing exception" << endl;
    throw runtime_error("Low-level error occurred!");
}

void midLevelOperation() {
    try {
        cout << "   midLevelOperation: Calling lowLevelOperation" << endl;
        lowLevelOperation();
    }
    catch (const exception& e) {
        cout << "   midLevelOperation: Caught exception: " << e.what() << endl;
        cout << "   midLevelOperation: Doing some cleanup..." << endl;
        // Do some cleanup or logging
        cout << "   midLevelOperation: Rethrowing exception" << endl;
        throw;  // Rethrow the same exception
    }
}

// Example 2: Conditional rethrow
void processTransaction(int amount) {
    try {
        if (amount < 0) {
            throw invalid_argument("Amount cannot be negative");
        }
        if (amount > 10000) {
            throw runtime_error("Amount exceeds limit");
        }
        cout << "   Transaction processed: $" << amount << endl;
    }
    catch (const invalid_argument& e) {
        // Handle this specific error locally
        cout << "   Validation error handled: " << e.what() << endl;
        // Don't rethrow - error is handled
    }
    catch (const runtime_error& e) {
        // This error is too serious - rethrow it
        cout << "   Critical error - cannot handle locally: " << e.what() << endl;
        throw;  // Rethrow
    }
}

// Example 3: Exception wrapping
void databaseQuery(const string& query) {
    if (query.empty()) {
        throw runtime_error("Empty query string");
    }
    if (query.find("DROP") != string::npos) {
        throw runtime_error("Dangerous query detected");
    }
    cout << "   Query executed: " << query << endl;
}

void executeQuery(const string& query) {
    try {
        databaseQuery(query);
    }
    catch (const exception& e) {
        // Wrap the original exception with more context
        string newMsg = "Query execution failed: " + string(e.what());
        throw WrappedException(newMsg, e.what());
    }
}

// Example 4: Resource cleanup with rethrow
class Resource {
private:
    string name;
public:
    Resource(const string& n) : name(n) {
        cout << "   [Resource " << name << " acquired]" << endl;
    }
    ~Resource() {
        cout << "   [Resource " << name << " released]" << endl;
    }
};

void operationWithResource() {
    Resource res("File Handle");

    try {
        cout << "   Performing operation..." << endl;
        throw runtime_error("Operation failed!");
    }
    catch (const exception& e) {
        cout << "   Caught exception: " << e.what() << endl;
        cout << "   Cleaning up..." << endl;
        // Resource will be automatically cleaned up
        throw;  // Rethrow after cleanup
    }
}

// Example 5: Nested exception handling (C++11)
void nestedExceptionExample() {
    try {
        try {
            throw runtime_error("Original error");
        }
        catch (const exception& e) {
            cout << "   Inner catch: " << e.what() << endl;
            // Throw new exception while preserving the original
            throw_with_nested(logic_error("Wrapped error"));
        }
    }
    catch (const logic_error& e) {
        cout << "   Outer catch: " << e.what() << endl;

        // Try to access nested exception
        try {
            rethrow_if_nested(e);
        }
        catch (const runtime_error& nested) {
            cout << "   Nested exception: " << nested.what() << endl;
        }
    }
}

int main() {
    cout << "=== Rethrowing Exceptions ===" << endl;

    // Example 1: Simple rethrow
    cout << "\n1. Simple Rethrow (logging and cleanup):" << endl;
    try {
        midLevelOperation();
    }
    catch (const exception& e) {
        cout << "   main: Caught rethrown exception: " << e.what() << endl;
    }

    // Example 2: Conditional rethrow
    cout << "\n2. Conditional Rethrow:" << endl;

    cout << "\n   a) Valid amount:" << endl;
    try {
        processTransaction(100);
    }
    catch (const exception& e) {
        cout << "   main: Caught exception: " << e.what() << endl;
    }

    cout << "\n   b) Invalid amount (handled locally):" << endl;
    try {
        processTransaction(-50);
    }
    catch (const exception& e) {
        cout << "   main: Caught exception: " << e.what() << endl;
    }

    cout << "\n   c) Amount too large (rethrown):" << endl;
    try {
        processTransaction(15000);
    }
    catch (const exception& e) {
        cout << "   main: Caught rethrown exception: " << e.what() << endl;
    }

    // Example 3: Exception wrapping
    cout << "\n3. Exception Wrapping:" << endl;
    try {
        executeQuery("DROP TABLE users");
    }
    catch (const WrappedException& e) {
        cout << "   Caught wrapped exception:" << endl;
        cout << "   - New message: " << e.what() << endl;
        cout << "   - Original message: " << e.getOriginalMessage() << endl;
    }

    // Example 4: Resource cleanup with rethrow
    cout << "\n4. Resource Cleanup with Rethrow:" << endl;
    try {
        operationWithResource();
    }
    catch (const exception& e) {
        cout << "   main: Final handler: " << e.what() << endl;
    }

    // Example 5: Nested exceptions
    cout << "\n5. Nested Exceptions (C++11):" << endl;
    nestedExceptionExample();

    // Example 6: throw vs throw e
    cout << "\n6. Difference: throw; vs throw e;" << endl;
    cout << "   Note: 'throw;' preserves the original exception type" << endl;
    cout << "   Note: 'throw e;' may cause object slicing with derived types" << endl;

    try {
        try {
            throw out_of_range("Original out_of_range exception");
        }
        catch (const exception& e) {
            cout << "   Rethrowing with 'throw;'" << endl;
            throw;  // Correct: preserves out_of_range type
        }
    }
    catch (const out_of_range& e) {
        cout << "   Caught as out_of_range: " << e.what() << endl;
    }
    catch (const exception& e) {
        cout << "   Caught as generic exception: " << e.what() << endl;
    }

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. Use 'throw;' to rethrow the current exception
     * 2. Rethrow preserves the original exception type
     * 3. Use rethrow for partial handling (cleanup + propagate)
     * 4. NEVER use 'throw e;' - it causes object slicing
     * 5. Rethrow allows intermediate error logging
     * 6. Can wrap exceptions to add context
     * 7. Cleanup happens before rethrow (RAII helps)
     * 8. Conditional rethrow based on error severity
     * 9. throw_with_nested for exception chaining (C++11)
     * 10. Only rethrow inside catch block
     */

    return 0;
}
