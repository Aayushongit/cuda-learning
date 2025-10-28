/**
 * 09_noexcept.cpp
 *
 * TOPIC: noexcept Specifier and Exception Specifications
 *
 * This file demonstrates:
 * - noexcept specifier
 * - Conditional noexcept
 * - noexcept operator (compile-time check)
 * - Performance benefits of noexcept
 * - When to use noexcept
 * - Difference from old throw() specification
 */

#include <iostream>
#include <stdexcept>
#include <vector>
#include <type_traits>

using namespace std;

/**
 * noexcept SPECIFIER:
 * - Declares that a function never throws exceptions
 * - If exception is thrown, std::terminate() is called
 * - Allows compiler optimizations
 * - Important for move operations and STL efficiency
 */

// Example 1: Basic noexcept
void functionThatNeverThrows() noexcept {
    cout << "   This function is guaranteed not to throw" << endl;
    // Can't throw - will call std::terminate if it does
}

void functionThatMightThrow() {
    cout << "   This function might throw" << endl;
    // Could throw exceptions
}

void functionThatLies() noexcept {
    cout << "   This says noexcept but..." << endl;
    // throw runtime_error("Oops!");  // Calls std::terminate()!
}

// Example 2: Conditional noexcept
template<typename T>
void swap_values(T& a, T& b) noexcept(is_nothrow_move_constructible<T>::value &&
                                       is_nothrow_move_assignable<T>::value) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

// Example 3: noexcept operator (compile-time check)
void demonstrateNoexceptOperator() {
    cout << "\n3. noexcept Operator (Compile-time Check):" << endl;

    // noexcept(expression) returns true if expression doesn't throw
    cout << "   functionThatNeverThrows is noexcept: "
         << noexcept(functionThatNeverThrows()) << endl;

    cout << "   functionThatMightThrow is noexcept: "
         << noexcept(functionThatMightThrow()) << endl;

    // Checking expressions
    int x = 5, y = 10;
    cout << "   x + y is noexcept: " << noexcept(x + y) << endl;

    vector<int> vec;
    cout << "   vec.push_back(1) is noexcept: "
         << noexcept(vec.push_back(1)) << endl;

    cout << "   vec.clear() is noexcept: "
         << noexcept(vec.clear()) << endl;
}

// Example 4: noexcept in classes
class Widget {
private:
    int* data;
    size_t size;

public:
    // Constructor can throw
    Widget(size_t s) : data(new int[s]), size(s) {
        cout << "   Widget constructed" << endl;
    }

    // Destructor should always be noexcept (default in C++11)
    ~Widget() noexcept {
        delete[] data;
        cout << "   Widget destroyed" << endl;
    }

    // Move constructor - noexcept is crucial for vector optimization
    Widget(Widget&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        cout << "   Widget move constructed (noexcept)" << endl;
    }

    // Move assignment - should be noexcept
    Widget& operator=(Widget&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
            cout << "   Widget move assigned (noexcept)" << endl;
        }
        return *this;
    }

    // Copy operations (can throw)
    Widget(const Widget& other) : data(new int[other.size]), size(other.size) {
        cout << "   Widget copy constructed (can throw)" << endl;
        for (size_t i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
    }

    Widget& operator=(const Widget& other) {
        if (this != &other) {
            int* newData = new int[other.size];  // Can throw
            delete[] data;
            data = newData;
            size = other.size;
            for (size_t i = 0; i < size; i++) {
                data[i] = other.data[i];
            }
            cout << "   Widget copy assigned (can throw)" << endl;
        }
        return *this;
    }

    // Swap should be noexcept
    void swap(Widget& other) noexcept {
        std::swap(data, other.data);
        std::swap(size, other.size);
    }

    size_t getSize() const noexcept { return size; }
};

// Example 5: noexcept and std::vector optimization
void vectorOptimization() {
    cout << "\n5. Vector Optimization with noexcept:" << endl;

    cout << "   Widget move constructor is noexcept: "
         << is_nothrow_move_constructible<Widget>::value << endl;

    vector<Widget> widgets;
    widgets.reserve(2);

    widgets.push_back(Widget(10));
    cout << "   Vector size: " << widgets.size() << endl;

    widgets.push_back(Widget(20));
    cout << "   Vector size: " << widgets.size() << endl;

    // This will cause reallocation
    cout << "   Adding third widget (causes reallocation):" << endl;
    widgets.push_back(Widget(30));
    cout << "   Vector size: " << widgets.size() << endl;
    cout << "   Note: If move constructor is noexcept, vector uses move" << endl;
    cout << "         Otherwise, it copies for exception safety!" << endl;
}

// Example 6: Functions that should be noexcept
class BestPractices {
public:
    // Destructors (implicitly noexcept in C++11)
    ~BestPractices() noexcept {
        // Never throw from destructor
    }

    // Move operations
    BestPractices(BestPractices&& other) noexcept {
        // Move should not throw
    }

    BestPractices& operator=(BestPractices&& other) noexcept {
        // Move assignment should not throw
        return *this;
    }

    // Swap
    void swap(BestPractices& other) noexcept {
        // Swap should not throw
    }

    // Memory deallocation
    void deallocate() noexcept {
        // Cleanup should not throw
    }

    // Default constructor (if possible)
    BestPractices() noexcept {
        // Simple initialization
    }
};

// Example 7: Wide vs Narrow Contracts
// Narrow contract: has preconditions, can't be noexcept
int divide(int a, int b) {
    if (b == 0) {
        throw invalid_argument("Division by zero");
    }
    return a / b;
}

// Wide contract: no preconditions, can be noexcept
int multiply(int a, int b) noexcept {
    return a * b;
}

// Example 8: Conditional noexcept with templates
template<typename T>
class Container {
private:
    T* data;
    size_t size;

public:
    // noexcept depends on T's move constructor
    Container(Container&& other) noexcept(is_nothrow_move_constructible<T>::value)
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    // noexcept depends on T's destructor
    ~Container() noexcept(is_nothrow_destructible<T>::value) {
        delete[] data;
    }
};

int main() {
    cout << "=== noexcept Specifier ===" << endl;

    // Example 1: Basic noexcept
    cout << "\n1. Basic noexcept Functions:" << endl;
    functionThatNeverThrows();
    functionThatMightThrow();
    functionThatLies();  // Safe unless we uncomment the throw

    // Example 2: Conditional noexcept
    cout << "\n2. Conditional noexcept:" << endl;
    int a = 5, b = 10;
    cout << "   Before swap: a=" << a << ", b=" << b << endl;
    swap_values(a, b);
    cout << "   After swap: a=" << a << ", b=" << b << endl;
    cout << "   swap_values<int> is noexcept: "
         << noexcept(swap_values(a, b)) << endl;

    // Example 3: noexcept operator
    demonstrateNoexceptOperator();

    // Example 4: noexcept in classes
    cout << "\n4. noexcept in Classes:" << endl;
    Widget w1(100);
    Widget w2(200);

    cout << "   Widget move constructor is noexcept: "
         << noexcept(Widget(std::move(w1))) << endl;

    cout << "   Widget destructor is noexcept: "
         << noexcept(w1.~Widget()) << endl;

    // Example 5: Vector optimization
    vectorOptimization();

    // Example 7: Wide vs Narrow contracts
    cout << "\n6. Wide vs Narrow Contracts:" << endl;
    cout << "   multiply(5, 4) is noexcept: "
         << noexcept(multiply(5, 4)) << endl;
    cout << "   divide(10, 2) is noexcept: "
         << noexcept(divide(10, 2)) << endl;

    // Example 8: Performance impact
    cout << "\n7. Performance Impact:" << endl;
    cout << "   noexcept allows compiler optimizations:" << endl;
    cout << "   - No exception handling code generated" << endl;
    cout << "   - Better inlining opportunities" << endl;
    cout << "   - Vector can use move instead of copy" << endl;
    cout << "   - Crucial for move operations and algorithms" << endl;

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. noexcept means function won't throw exceptions
     * 2. If noexcept function throws, std::terminate() called
     * 3. noexcept(expression) checks if expression can throw
     * 4. Conditional noexcept: noexcept(condition)
     * 5. Destructors are implicitly noexcept in C++11
     * 6. Move operations should be noexcept for efficiency
     * 7. Vector uses move if noexcept, copy otherwise
     * 8. Swap should always be noexcept
     * 9. noexcept enables compiler optimizations
     * 10. Use noexcept for wide contracts, avoid for narrow
     *
     * WHEN TO USE noexcept:
     * - Destructors (automatic)
     * - Move constructors and move assignment
     * - Swap functions
     * - Memory deallocation functions
     * - Default constructors (if simple)
     * - Functions with wide contracts
     * - Leaf functions that don't call throwing functions
     *
     * WHEN NOT TO USE noexcept:
     * - Functions that validate input (narrow contracts)
     * - Functions that call potentially throwing functions
     * - When unsure (better to not use than to lie)
     */

    return 0;
}
