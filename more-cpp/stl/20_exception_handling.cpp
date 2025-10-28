/**
 * 20_exception_handling.cpp
 *
 * EXCEPTION HANDLING
 * - try-catch blocks
 * - Standard exceptions
 * - Custom exceptions
 * - Exception safety guarantees
 * - noexcept specifier
 */

#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

// Template function for conditional noexcept demonstration
template<typename T>
void swap_values(T& a, T& b) noexcept(std::is_nothrow_move_constructible_v<T>) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

int main() {
    std::cout << "=== EXCEPTION HANDLING ===\n";

    separator("BASIC EXCEPTION HANDLING");

    // 1. Basic try-catch
    std::cout << "\n1. BASIC TRY-CATCH:\n";
    try {
        throw std::runtime_error("Something went wrong!");
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
    }

    // 2. Multiple catch blocks
    std::cout << "\n2. MULTIPLE CATCH BLOCKS:\n";
    try {
        // throw std::runtime_error("Runtime error");
        // throw std::logic_error("Logic error");
        throw 42;
    } catch (const std::runtime_error& e) {
        std::cout << "Runtime error: " << e.what() << "\n";
    } catch (const std::logic_error& e) {
        std::cout << "Logic error: " << e.what() << "\n";
    } catch (int num) {
        std::cout << "Caught int: " << num << "\n";
    } catch (...) {
        std::cout << "Caught unknown exception\n";
    }

    // 3. Catching by Reference
    std::cout << "\n3. CATCH BY REFERENCE:\n";
    try {
        throw std::invalid_argument("Bad argument");
    } catch (const std::exception& e) {  // Catch base class by reference
        std::cout << "Caught: " << e.what() << "\n";
    }

    separator("STANDARD EXCEPTIONS");

    // 4. logic_error family
    std::cout << "\n4. LOGIC_ERROR FAMILY:\n";
    try {
        throw std::invalid_argument("Invalid argument provided");
    } catch (const std::logic_error& e) {
        std::cout << "Logic error: " << e.what() << "\n";
    }

    try {
        throw std::out_of_range("Index out of range");
    } catch (const std::out_of_range& e) {
        std::cout << "Out of range: " << e.what() << "\n";
    }

    // 5. runtime_error family
    std::cout << "\n5. RUNTIME_ERROR FAMILY:\n";
    try {
        throw std::overflow_error("Arithmetic overflow");
    } catch (const std::runtime_error& e) {
        std::cout << "Runtime error: " << e.what() << "\n";
    }

    try {
        throw std::underflow_error("Arithmetic underflow");
    } catch (const std::underflow_error& e) {
        std::cout << "Underflow: " << e.what() << "\n";
    }

    // 6. bad_alloc
    std::cout << "\n6. BAD_ALLOC:\n";
    try {
        // Try to allocate huge amount
        // std::vector<int> v(std::numeric_limits<size_t>::max());
        throw std::bad_alloc();
    } catch (const std::bad_alloc& e) {
        std::cout << "Allocation failed: " << e.what() << "\n";
    }

    separator("CUSTOM EXCEPTIONS");

    // 7. Custom Exception Class
    std::cout << "\n7. CUSTOM EXCEPTION:\n";
    class MyException : public std::exception {
    private:
        std::string message;
    public:
        explicit MyException(const std::string& msg) : message(msg) {}
        const char* what() const noexcept override {
            return message.c_str();
        }
    };

    try {
        throw MyException("Custom error occurred");
    } catch (const MyException& e) {
        std::cout << "Caught custom exception: " << e.what() << "\n";
    }

    // 8. Exception with Data
    std::cout << "\n8. EXCEPTION WITH DATA:\n";
    class FileException : public std::exception {
    private:
        std::string filename;
        int error_code;
        mutable std::string full_message;
    public:
        FileException(const std::string& file, int code)
            : filename(file), error_code(code) {}

        const char* what() const noexcept override {
            full_message = "File error in '" + filename + "' (code: " + std::to_string(error_code) + ")";
            return full_message.c_str();
        }

        const std::string& get_filename() const { return filename; }
        int get_error_code() const { return error_code; }
    };

    try {
        throw FileException("data.txt", 404);
    } catch (const FileException& e) {
        std::cout << e.what() << "\n";
        std::cout << "File: " << e.get_filename() << ", Code: " << e.get_error_code() << "\n";
    }

    separator("EXCEPTION PROPAGATION");

    // 9. Rethrowing Exceptions
    std::cout << "\n9. RETHROWING:\n";
    auto process = []() {
        try {
            throw std::runtime_error("Error in process");
        } catch (const std::exception& e) {
            std::cout << "Caught in process, rethrowing...\n";
            throw;  // Rethrow same exception
        }
    };

    try {
        process();
    } catch (const std::exception& e) {
        std::cout << "Caught rethrown exception: " << e.what() << "\n";
    }

    // 10. Nested Exceptions (C++11)
    std::cout << "\n10. NESTED EXCEPTIONS:\n";
    try {
        try {
            throw std::runtime_error("Original error");
        } catch (...) {
            std::throw_with_nested(std::runtime_error("Wrapped error"));
        }
    } catch (const std::exception& e) {
        std::cout << "Outer exception: " << e.what() << "\n";
        try {
            std::rethrow_if_nested(e);
        } catch (const std::exception& nested) {
            std::cout << "  Nested exception: " << nested.what() << "\n";
        }
    }

    separator("RAII AND EXCEPTION SAFETY");

    // 11. RAII Example
    std::cout << "\n11. RAII (Exception-Safe Resource Management):\n";
    class Resource {
    public:
        Resource() { std::cout << "Resource acquired\n"; }
        ~Resource() { std::cout << "Resource released\n"; }
    };

    try {
        Resource r;
        throw std::runtime_error("Error occurred");
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }
    // Resource automatically cleaned up

    // 12. Smart Pointers for Exception Safety
    std::cout << "\n12. SMART POINTERS (EXCEPTION-SAFE):\n";
    try {
        auto ptr = std::make_unique<int>(42);
        std::cout << "Value: " << *ptr << "\n";
        throw std::runtime_error("Error");
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }
    // Smart pointer automatically cleaned up

    separator("NOEXCEPT");

    // 13. noexcept Specifier
    std::cout << "\n13. NOEXCEPT:\n";
    auto no_throw_func = []() noexcept {
        return 42;
    };

    auto may_throw_func = []() {
        throw std::runtime_error("May throw");
    };

    std::cout << "no_throw_func is noexcept: " << noexcept(no_throw_func()) << "\n";
    std::cout << "may_throw_func is noexcept: " << noexcept(may_throw_func()) << "\n";

    // 14. Conditional noexcept
    std::cout << "\n14. CONDITIONAL NOEXCEPT:\n";
    int x = 1, y = 2;
    swap_values(x, y);
    std::cout << "Swapped: x=" << x << ", y=" << y << "\n";

    separator("EXCEPTION SAFETY GUARANTEES");

    std::cout << "\n15. EXCEPTION SAFETY GUARANTEES:\n\n";

    std::cout << "1. NO-THROW GUARANTEE:\n";
    std::cout << "   - Function never throws\n";
    std::cout << "   - Mark with noexcept\n";
    std::cout << "   - Example: destructors, move operations\n\n";

    std::cout << "2. STRONG GUARANTEE (commit-or-rollback):\n";
    std::cout << "   - If exception thrown, state unchanged\n";
    std::cout << "   - Example: vector::push_back\n\n";

    std::cout << "3. BASIC GUARANTEE:\n";
    std::cout << "   - If exception thrown, no resources leaked\n";
    std::cout << "   - Object may be in valid but undefined state\n\n";

    std::cout << "4. NO GUARANTEE:\n";
    std::cout << "   - Resource leaks or invalid state possible\n";
    std::cout << "   - Avoid this!\n";

    separator("BEST PRACTICES");

    std::cout << "\n1. Catch exceptions by const reference\n";
    std::cout << "2. Use RAII for resource management\n";
    std::cout << "3. Don't throw in destructors\n";
    std::cout << "4. Don't throw in noexcept functions\n";
    std::cout << "5. Catch specific exceptions before general ones\n";
    std::cout << "6. Use standard exception types when possible\n";
    std::cout << "7. Provide strong exception guarantee when possible\n";
    std::cout << "8. Document exception behavior\n";
    std::cout << "9. Use exceptions for exceptional situations, not control flow\n";
    std::cout << "10. Consider performance cost of exceptions\n";

    separator("WHEN TO USE EXCEPTIONS");

    std::cout << "\nUse exceptions for:\n";
    std::cout << "- Errors that can't be handled locally\n";
    std::cout << "- Constructor failures\n";
    std::cout << "- Error handling across multiple layers\n";
    std::cout << "- Unrecoverable errors\n";

    std::cout << "\nDon't use exceptions for:\n";
    std::cout << "- Expected/common errors (use error codes)\n";
    std::cout << "- Performance-critical paths\n";
    std::cout << "- Flow control\n";
    std::cout << "- Cases where std::optional or std::expected work\n";

    std::cout << "\n=== END OF EXCEPTION HANDLING ===\n";

    return 0;
}
