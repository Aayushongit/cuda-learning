/**
 * 06_nested_try_catch.cpp
 *
 * TOPIC: Nested Try-Catch Blocks
 *
 * This file demonstrates:
 * - Nested try-catch blocks
 * - Exception propagation through nested scopes
 * - Function call stack unwinding
 * - Multiple levels of exception handling
 * - When to use nested try-catch
 */

#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

// Example 1: Basic nested try-catch
void nestedTryCatchExample() {
    cout << "   Entering outer try block" << endl;

    try {
        cout << "   Outer try: About to enter inner try" << endl;

        try {
            cout << "   Inner try: Throwing exception" << endl;
            throw runtime_error("Inner exception");
            cout << "   This line never executes" << endl;
        }
        catch (const invalid_argument& e) {
            // This won't catch runtime_error
            cout << "   Inner catch (invalid_argument): " << e.what() << endl;
        }

        // If inner catch doesn't handle it, exception propagates to outer catch
        cout << "   This line won't execute if exception wasn't caught" << endl;
    }
    catch (const runtime_error& e) {
        cout << "   Outer catch: Caught runtime_error: " << e.what() << endl;
    }

    cout << "   After try-catch blocks - execution continues" << endl;
}

// Example 2: Different exception handling at different levels
void processData(const string& data) {
    try {
        if (data.empty()) {
            throw invalid_argument("Data is empty");
        }

        try {
            if (data.length() < 5) {
                throw length_error("Data too short");
            }

            if (data == "ERROR") {
                throw runtime_error("Invalid data content");
            }

            cout << "   Data processed successfully: " << data << endl;
        }
        catch (const length_error& e) {
            // Handle length errors specifically at this level
            cout << "   Inner handler: " << e.what()
                 << " - using default values" << endl;
            // Continue processing with defaults
        }

        cout << "   Final processing complete" << endl;
    }
    catch (const invalid_argument& e) {
        // Handle validation errors at outer level
        cout << "   Outer handler: " << e.what() << endl;
    }
    // runtime_error is not caught - will propagate to caller
}

// Example 3: Exception unwinding through function calls
void level3Function() {
    cout << "      Level 3: Throwing exception" << endl;
    throw runtime_error("Error at level 3");
}

void level2Function() {
    cout << "    Level 2: Calling level 3" << endl;
    try {
        level3Function();
        cout << "    Level 2: After level 3 call (not reached)" << endl;
    }
    catch (const logic_error& e) {
        // Won't catch runtime_error
        cout << "    Level 2: Caught logic_error" << endl;
    }
    cout << "    Level 2: Exiting (exception propagates)" << endl;
}

void level1Function() {
    cout << "  Level 1: Calling level 2" << endl;
    try {
        level2Function();
        cout << "  Level 1: After level 2 call (not reached)" << endl;
    }
    catch (const runtime_error& e) {
        cout << "  Level 1: Caught exception: " << e.what() << endl;
    }
    cout << "  Level 1: Exiting normally" << endl;
}

// Example 4: Resource management with nested try-catch
class DatabaseConnection {
private:
    string name;
public:
    DatabaseConnection(const string& n) : name(n) {
        cout << "   [DB Connection '" << name << "' opened]" << endl;
    }
    ~DatabaseConnection() {
        cout << "   [DB Connection '" << name << "' closed]" << endl;
    }
    void executeQuery(const string& query) {
        if (query.find("DELETE") != string::npos) {
            throw runtime_error("DELETE operation not allowed");
        }
        cout << "   Query executed: " << query << endl;
    }
};

void performDatabaseOperations() {
    DatabaseConnection db("MainDB");

    try {
        cout << "   Starting transaction..." << endl;

        try {
            db.executeQuery("SELECT * FROM users");

            try {
                db.executeQuery("DELETE FROM users");  // This will throw
            }
            catch (const runtime_error& e) {
                cout << "   Level 3 catch: Query failed: " << e.what() << endl;
                throw;  // Rethrow to rollback transaction
            }

            db.executeQuery("UPDATE users SET active=1");
        }
        catch (const exception& e) {
            cout << "   Level 2 catch: Rolling back transaction: " << e.what() << endl;
            throw;  // Rethrow to outer handler
        }

        cout << "   Committing transaction..." << endl;
    }
    catch (const exception& e) {
        cout << "   Level 1 catch: Transaction failed: " << e.what() << endl;
    }
    // DatabaseConnection destructor called here
}

// Example 5: Selective exception handling
void selectiveHandling(int scenario) {
    try {
        cout << "   Outer try block" << endl;

        try {
            cout << "   Inner try block - scenario " << scenario << endl;

            switch (scenario) {
                case 1:
                    throw invalid_argument("Invalid arg");
                case 2:
                    throw runtime_error("Runtime error");
                case 3:
                    throw logic_error("Logic error");
                default:
                    cout << "   No exception thrown" << endl;
            }
        }
        catch (const invalid_argument& e) {
            // Only handle this type at inner level
            cout << "   Inner catch: " << e.what() << " - handled locally" << endl;
            return;  // Exit function - exception handled
        }
        // Other exceptions propagate to outer catch

        cout << "   Between try-catch blocks" << endl;
    }
    catch (const exception& e) {
        // Catch everything else
        cout << "   Outer catch: " << e.what() << endl;
    }
}

int main() {
    cout << "=== Nested Try-Catch Blocks ===" << endl;

    // Example 1: Basic nested try-catch
    cout << "\n1. Basic Nested Try-Catch:" << endl;
    nestedTryCatchExample();

    // Example 2: Different handling levels
    cout << "\n2. Different Exception Handling at Different Levels:" << endl;

    cout << "\n   a) Empty data:" << endl;
    processData("");

    cout << "\n   b) Short data:" << endl;
    processData("Hi");

    cout << "\n   c) Valid data:" << endl;
    processData("Hello World");

    cout << "\n   d) Invalid content:" << endl;
    try {
        processData("ERROR");
    }
    catch (const runtime_error& e) {
        cout << "   main: Uncaught exception propagated: " << e.what() << endl;
    }

    // Example 3: Stack unwinding
    cout << "\n3. Exception Propagation Through Call Stack:" << endl;
    level1Function();

    // Example 4: Resource management
    cout << "\n4. Resource Management with Nested Try-Catch:" << endl;
    performDatabaseOperations();

    // Example 5: Selective handling
    cout << "\n5. Selective Exception Handling:" << endl;

    for (int i = 1; i <= 4; i++) {
        cout << "\n   Scenario " << i << ":" << endl;
        selectiveHandling(i);
    }

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. Nested try-catch allows granular error handling
     * 2. Unhandled exceptions propagate to outer catch blocks
     * 3. Exception propagates up the call stack until caught
     * 4. Each catch block can handle specific exception types
     * 5. Inner catch can handle some errors, let others propagate
     * 6. Resources are cleaned up during stack unwinding (RAII)
     * 7. Use nested try-catch for different error handling strategies
     * 8. Inner catch can log, cleanup, then rethrow
     * 9. Outer catch acts as safety net for unhandled exceptions
     * 10. Stack unwinding calls destructors of all local objects
     */

    return 0;
}
