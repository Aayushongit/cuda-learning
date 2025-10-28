/**
 * 07_constructor_exceptions.cpp
 *
 * TOPIC: Exception Handling in Constructors and Destructors
 *
 * This file demonstrates:
 * - Throwing exceptions from constructors
 * - Why destructors should never throw
 * - Initialization list exceptions
 * - Partially constructed objects
 * - Function-try-block for constructors
 * - Best practices for constructor exceptions
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>

using namespace std;

// Example 1: Basic constructor exception
class File {
private:
    string filename;
    bool isOpen;

public:
    File(const string& name) : filename(name), isOpen(false) {
        cout << "   Constructing File: " << filename << endl;

        if (filename.empty()) {
            throw invalid_argument("Filename cannot be empty");
        }

        if (filename == "invalid.txt") {
            throw runtime_error("Cannot open file: " + filename);
        }

        isOpen = true;
        cout << "   File opened successfully: " << filename << endl;
    }

    ~File() {
        cout << "   Destroying File: " << filename << endl;
        // Destructor should NOT throw exceptions
        if (isOpen) {
            cout << "   Closing file: " << filename << endl;
        }
    }

    string getName() const { return filename; }
};

// Example 2: Initialization list exceptions
class Resource {
private:
    string name;
public:
    Resource(const string& n) : name(n) {
        cout << "      [Resource '" << name << "' created]" << endl;
        if (name == "bad_resource") {
            throw runtime_error("Bad resource name");
        }
    }
    ~Resource() {
        cout << "      [Resource '" << name << "' destroyed]" << endl;
    }
};

class Manager {
private:
    Resource res1;
    Resource res2;
    int* data;

public:
    // Constructor with initialization list
    Manager(const string& r1, const string& r2)
        : res1(r1), res2(r2), data(nullptr)  // res1 and res2 constructed here
    {
        cout << "   Manager constructor body" << endl;

        // If exception thrown here, res1 and res2 are already constructed
        // Their destructors WILL be called
        data = new int[100];

        if (r2 == "trigger_error") {
            throw logic_error("Error in constructor body");
        }

        cout << "   Manager constructed successfully" << endl;
    }

    ~Manager() {
        cout << "   Manager destructor" << endl;
        delete[] data;
    }
};

// Example 3: Function-try-block for constructors
class SafeManager {
private:
    Resource resource;
    int* buffer;

public:
    // Function-try-block catches exceptions from initialization list
    SafeManager(const string& resName)
    try : resource(resName), buffer(new int[50])
    {
        cout << "   SafeManager constructor body" << endl;
        // Constructor body
    }
    catch (const exception& e) {
        // Catches exceptions from initialization list AND body
        cout << "   SafeManager constructor caught: " << e.what() << endl;
        // Note: buffer already allocated if exception in body
        // BUT member destructors NOT called yet
        // Exception is ALWAYS rethrown from here (implicitly or explicitly)
        throw;  // Rethrow (this happens automatically even if omitted)
    }

    ~SafeManager() {
        cout << "   SafeManager destructor" << endl;
        delete[] buffer;
    }
};

// Example 4: Avoiding constructor exceptions with two-phase initialization
class Database {
private:
    string connectionString;
    bool connected;

public:
    // Constructor doesn't throw - just initializes
    Database(const string& connStr)
        : connectionString(connStr), connected(false) {
        cout << "   Database object created (not connected)" << endl;
    }

    // Separate initialization function that can throw
    bool connect() {
        cout << "   Attempting to connect..." << endl;

        if (connectionString.empty()) {
            throw invalid_argument("Empty connection string");
        }

        if (connectionString == "bad_server") {
            throw runtime_error("Cannot connect to server");
        }

        connected = true;
        cout << "   Connected successfully" << endl;
        return true;
    }

    bool isConnected() const { return connected; }

    ~Database() {
        cout << "   Database destructor" << endl;
        if (connected) {
            cout << "   Disconnecting..." << endl;
        }
    }
};

// Example 5: Why destructors should NOT throw
class BadDestructor {
public:
    ~BadDestructor() noexcept(false) {  // Explicitly allow throwing (BAD PRACTICE!)
        cout << "   BadDestructor: Throwing in destructor (DON'T DO THIS!)" << endl;
        // throw runtime_error("Exception in destructor!");  // DANGEROUS!
    }
};

// Example 6: Smart pointers help with exception safety
class SmartManager {
private:
    unique_ptr<Resource> res1;
    unique_ptr<Resource> res2;
    unique_ptr<int[]> data;

public:
    SmartManager(const string& r1, const string& r2) {
        cout << "   Creating resources with smart pointers..." << endl;

        res1 = make_unique<Resource>(r1);
        // If next line throws, res1 is automatically cleaned up
        res2 = make_unique<Resource>(r2);
        // If next line throws, res1 and res2 are automatically cleaned up
        data = make_unique<int[]>(100);

        cout << "   SmartManager constructed successfully" << endl;
    }

    ~SmartManager() {
        cout << "   SmartManager destructor (smart pointers auto-cleanup)" << endl;
    }
};

int main() {
    cout << "=== Constructor and Destructor Exceptions ===" << endl;

    // Example 1: Basic constructor exception
    cout << "\n1. Constructor Exception:" << endl;

    cout << "\n   a) Valid file:" << endl;
    try {
        File f1("data.txt");
        cout << "   File created: " << f1.getName() << endl;
    }
    catch (const exception& e) {
        cout << "   Caught: " << e.what() << endl;
    }

    cout << "\n   b) Invalid file (constructor throws):" << endl;
    try {
        File f2("invalid.txt");
        cout << "   This won't execute" << endl;
    }
    catch (const exception& e) {
        cout << "   Caught: " << e.what() << endl;
        cout << "   Note: Destructor was NOT called (object not fully constructed)" << endl;
    }

    // Example 2: Initialization list exceptions
    cout << "\n2. Initialization List Exceptions:" << endl;

    cout << "\n   a) Success case:" << endl;
    try {
        Manager m1("resource1", "resource2");
    }
    catch (const exception& e) {
        cout << "   Caught: " << e.what() << endl;
    }

    cout << "\n   b) Exception in constructor body:" << endl;
    try {
        Manager m2("resource1", "trigger_error");
    }
    catch (const exception& e) {
        cout << "   Caught: " << e.what() << endl;
        cout << "   Note: Member destructors were called!" << endl;
    }

    cout << "\n   c) Exception in initialization list:" << endl;
    try {
        Manager m3("resource1", "bad_resource");
    }
    catch (const exception& e) {
        cout << "   Caught: " << e.what() << endl;
        cout << "   Note: res1 destructor called, but Manager destructor was NOT" << endl;
    }

    // Example 3: Function-try-block
    cout << "\n3. Function-Try-Block:" << endl;
    try {
        SafeManager sm("bad_resource");
    }
    catch (const exception& e) {
        cout << "   main caught: " << e.what() << endl;
    }

    // Example 4: Two-phase initialization
    cout << "\n4. Two-Phase Initialization (avoiding constructor exceptions):" << endl;

    cout << "\n   a) Successful connection:" << endl;
    try {
        Database db1("server.example.com");
        db1.connect();
        cout << "   Database ready to use" << endl;
    }
    catch (const exception& e) {
        cout << "   Connection failed: " << e.what() << endl;
    }

    cout << "\n   b) Failed connection:" << endl;
    try {
        Database db2("bad_server");
        db2.connect();
    }
    catch (const exception& e) {
        cout << "   Connection failed: " << e.what() << endl;
        cout << "   Note: Database object still exists and can be destroyed safely" << endl;
    }

    // Example 6: Smart pointers for exception safety
    cout << "\n5. Smart Pointers for Exception Safety:" << endl;

    cout << "\n   a) Success:" << endl;
    try {
        SmartManager sm1("res1", "res2");
    }
    catch (const exception& e) {
        cout << "   Caught: " << e.what() << endl;
    }

    cout << "\n   b) Failure (automatic cleanup):" << endl;
    try {
        SmartManager sm2("res1", "bad_resource");
    }
    catch (const exception& e) {
        cout << "   Caught: " << e.what() << endl;
        cout << "   Note: Smart pointers automatically cleaned up!" << endl;
    }

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. Constructors CAN throw exceptions
     * 2. If constructor throws, destructor is NOT called
     * 3. Member destructors ARE called for fully constructed members
     * 4. Destructors should NEVER throw exceptions
     * 5. Use noexcept on destructors
     * 6. Initialization list exceptions: only constructed members cleaned
     * 7. Function-try-block can catch initialization list exceptions
     * 8. Two-phase initialization avoids constructor exceptions
     * 9. Smart pointers provide automatic exception safety
     * 10. RAII + smart pointers = exception-safe code
     *
     * BEST PRACTICES:
     * - Mark destructors noexcept (default in C++11)
     * - Never throw from destructors
     * - Use smart pointers for resource management
     * - Consider two-phase initialization for complex objects
     * - Use RAII for automatic cleanup
     */

    return 0;
}
