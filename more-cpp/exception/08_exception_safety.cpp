/**
 * 08_exception_safety.cpp
 *
 * TOPIC: RAII and Exception Safety Guarantees
 *
 * This file demonstrates:
 * - RAII (Resource Acquisition Is Initialization) pattern
 * - Exception safety guarantees (no-throw, strong, basic)
 * - Writing exception-safe code
 * - Smart pointers and automatic resource management
 * - Exception-safe containers
 * - Best practices for exception safety
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

using namespace std;

/**
 * EXCEPTION SAFETY GUARANTEES:
 *
 * 1. No-throw guarantee (noexcept):
 *    - Operation never throws exceptions
 *    - Examples: destructors, swap, move operations
 *
 * 2. Strong guarantee (commit-or-rollback):
 *    - Operation succeeds completely OR
 *    - Program state remains unchanged (as if operation never started)
 *
 * 3. Basic guarantee:
 *    - Operation might partially succeed
 *    - No resources leaked, objects in valid state
 *    - But state might be changed
 *
 * 4. No guarantee:
 *    - May leak resources or leave objects in invalid state
 *    - AVOID THIS!
 */

// Example 1: RAII Pattern - Manual Resource Management (BAD)
void badFileHandling() {
    cout << "   BAD: Manual resource management" << endl;

    FILE* file = fopen("data.txt", "r");
    if (!file) {
        throw runtime_error("Cannot open file");
    }

    // If an exception is thrown here, file is leaked!
    // processData(file);

    fclose(file);  // This line might never be reached!
}

// Example 1b: RAII Pattern - Automatic Resource Management (GOOD)
class FileHandle {
private:
    FILE* file;
    string filename;

public:
    FileHandle(const string& name, const string& mode) : file(nullptr), filename(name) {
        file = fopen(name.c_str(), mode.c_str());
        if (!file) {
            throw runtime_error("Cannot open file: " + name);
        }
        cout << "   [File opened: " << filename << "]" << endl;
    }

    ~FileHandle() {
        if (file) {
            fclose(file);
            cout << "   [File closed: " << filename << "]" << endl;
        }
    }

    // Delete copy operations to prevent double-close
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    FILE* get() const { return file; }
};

void goodFileHandling() {
    cout << "   GOOD: RAII - automatic resource management" << endl;

    FileHandle file("data.txt", "r");
    // File automatically closed when function exits (normal or exception)

    // Even if exception thrown here, destructor called
    // throw runtime_error("Some error");

    cout << "   File operations complete" << endl;
}  // File automatically closed here

// Example 2: Smart Pointers for Exception Safety
class Widget {
private:
    int id;
public:
    Widget(int i) : id(i) {
        cout << "   [Widget " << id << " created]" << endl;
        if (id == 666) {
            throw runtime_error("Evil widget number!");
        }
    }
    ~Widget() {
        cout << "   [Widget " << id << " destroyed]" << endl;
    }
    int getId() const { return id; }
};

void rawPointerExample() {
    cout << "   BAD: Using raw pointers" << endl;

    Widget* w1 = new Widget(1);

    try {
        Widget* w2 = new Widget(666);  // Throws!
        delete w2;
    }
    catch (...) {
        // w2 was never assigned, but w1 still exists
        cout << "   Exception caught, but w1 might leak!" << endl;
        delete w1;  // Must remember to delete
        throw;
    }

    delete w1;
}

void smartPointerExample() {
    cout << "   GOOD: Using smart pointers" << endl;

    unique_ptr<Widget> w1 = make_unique<Widget>(1);

    try {
        unique_ptr<Widget> w2 = make_unique<Widget>(666);  // Throws!
    }
    catch (...) {
        cout << "   Exception caught, w1 automatically cleaned up!" << endl;
        throw;
    }
    // w1 automatically deleted here
}

// Example 3: Exception-Safe Swap
class DataBuffer {
private:
    int* data;
    size_t size;

public:
    DataBuffer(size_t s) : data(new int[s]), size(s) {
        cout << "   Buffer of size " << size << " created" << endl;
    }

    ~DataBuffer() {
        delete[] data;
        cout << "   Buffer of size " << size << " destroyed" << endl;
    }

    // No-throw swap (critical for strong exception guarantee)
    void swap(DataBuffer& other) noexcept {
        std::swap(data, other.data);
        std::swap(size, other.size);
    }

    // Copy assignment with strong exception guarantee
    DataBuffer& operator=(const DataBuffer& other) {
        if (this != &other) {
            // Create temporary copy (might throw)
            DataBuffer temp(other.size);
            for (size_t i = 0; i < other.size; i++) {
                temp.data[i] = other.data[i];
            }

            // No-throw swap (commit the change)
            swap(temp);
            // Old data destroyed in temp's destructor
        }
        return *this;
    }

    size_t getSize() const { return size; }
};

// Example 4: Strong Exception Guarantee
class Account {
private:
    string owner;
    double balance;

public:
    Account(const string& name, double bal) : owner(name), balance(bal) {}

    // Strong guarantee: either succeeds or balance unchanged
    void transfer(Account& to, double amount) {
        if (amount <= 0) {
            throw invalid_argument("Amount must be positive");
        }
        if (balance < amount) {
            throw runtime_error("Insufficient funds");
        }

        // Make a copy of states
        double oldBalance = balance;
        double oldToBalance = to.balance;

        try {
            // Perform operations
            balance -= amount;

            // Simulate potential error
            if (to.owner == "BadAccount") {
                throw runtime_error("Cannot transfer to bad account");
            }

            to.balance += amount;
        }
        catch (...) {
            // Rollback changes
            balance = oldBalance;
            to.balance = oldToBalance;
            throw;  // Rethrow
        }
    }

    double getBalance() const { return balance; }
    string getOwner() const { return owner; }
};

// Example 5: Vector and Exception Safety
void vectorExceptionSafety() {
    cout << "   Vector automatic cleanup:" << endl;

    vector<unique_ptr<Widget>> widgets;

    try {
        widgets.push_back(make_unique<Widget>(1));
        widgets.push_back(make_unique<Widget>(2));
        widgets.push_back(make_unique<Widget>(3));

        cout << "   Vector has " << widgets.size() << " widgets" << endl;

        // Simulate error
        throw runtime_error("Something went wrong");
    }
    catch (...) {
        cout << "   Exception caught - vector will clean up all widgets" << endl;
        throw;
    }
    // Vector destructor automatically deletes all unique_ptrs
}

int main() {
    cout << "=== RAII and Exception Safety ===" << endl;

    // Example 1: RAII Pattern
    cout << "\n1. RAII Pattern:" << endl;

    try {
        goodFileHandling();
    }
    catch (const exception& e) {
        cout << "   Note: File automatically closed even though exception thrown" << endl;
    }

    // Example 2: Smart Pointers
    cout << "\n2. Smart Pointers for Automatic Cleanup:" << endl;

    try {
        smartPointerExample();
    }
    catch (const exception& e) {
        cout << "   All resources cleaned up automatically" << endl;
    }

    // Example 3: No-throw Swap
    cout << "\n3. No-throw Swap:" << endl;
    DataBuffer buf1(10);
    DataBuffer buf2(20);
    cout << "   Before swap: buf1=" << buf1.getSize()
         << ", buf2=" << buf2.getSize() << endl;
    buf1.swap(buf2);  // Never throws
    cout << "   After swap: buf1=" << buf1.getSize()
         << ", buf2=" << buf2.getSize() << endl;

    // Example 4: Strong Exception Guarantee
    cout << "\n4. Strong Exception Guarantee:" << endl;

    Account alice("Alice", 1000);
    Account bob("Bob", 500);

    cout << "   Before: Alice=$" << alice.getBalance()
         << ", Bob=$" << bob.getBalance() << endl;

    try {
        alice.transfer(bob, 200);
        cout << "   After successful transfer: Alice=$" << alice.getBalance()
             << ", Bob=$" << bob.getBalance() << endl;
    }
    catch (const exception& e) {
        cout << "   Transfer failed: " << e.what() << endl;
    }

    // Failed transfer
    Account charlie("Charlie", 100);
    Account badAccount("BadAccount", 0);

    cout << "\n   Before failed transfer: Charlie=$" << charlie.getBalance()
         << ", BadAccount=$" << badAccount.getBalance() << endl;

    try {
        charlie.transfer(badAccount, 50);
    }
    catch (const exception& e) {
        cout << "   Transfer failed: " << e.what() << endl;
        cout << "   After rollback: Charlie=$" << charlie.getBalance()
             << ", BadAccount=$" << badAccount.getBalance() << endl;
        cout << "   Note: Balances unchanged (strong guarantee)!" << endl;
    }

    // Example 5: Vector Exception Safety
    cout << "\n5. STL Container Exception Safety:" << endl;
    try {
        vectorExceptionSafety();
    }
    catch (const exception& e) {
        cout << "   All widgets automatically destroyed" << endl;
    }

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. RAII: Resource lifetime tied to object lifetime
     * 2. Use smart pointers (unique_ptr, shared_ptr)
     * 3. Destructors automatically called during unwinding
     * 4. No-throw guarantee for: destructors, swap, move ops
     * 5. Strong guarantee: commit-or-rollback semantics
     * 6. Use swap for strong guarantee in assignments
     * 7. STL containers provide exception safety
     * 8. Never use raw new/delete - use smart pointers
     * 9. Mark no-throw operations with noexcept
     * 10. Design for automatic cleanup
     *
     * BEST PRACTICES:
     * - Always use RAII for resource management
     * - Prefer smart pointers over raw pointers
     * - Implement no-throw swap
     * - Use copy-and-swap for strong guarantee
     * - Make destructors noexcept
     * - Use STL containers (exception-safe)
     * - Avoid manual memory management
     */

    return 0;
}
