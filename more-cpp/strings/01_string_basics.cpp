/**
 * 01_string_basics.cpp
 *
 * Introduction to C++ Strings
 *
 * WHAT IS A STRING?
 * - A string is a sequence of characters
 * - In C++, we use std::string class from <string> header
 * - Unlike C-style strings (char arrays), C++ strings are:
 *   * Dynamically sized (can grow/shrink)
 *   * Easier to use
 *   * Safer (automatic memory management)
 *   * Feature-rich (many built-in methods)
 */

#include <iostream>
#include <string>  // Required for std::string

int main() {
    std::cout << "=== C++ STRING BASICS ===" << std::endl;

    // 1. CREATING A STRING
    std::string greeting = "Hello, World!";
    std::cout << "\n1. Basic string: " << greeting << std::endl;

    // 2. EMPTY STRING
    std::string empty;
    std::cout << "2. Empty string (length): " << empty.length() << std::endl;

    // 3. STRING LENGTH
    std::cout << "3. Greeting length: " << greeting.length() << " characters" << std::endl;
    std::cout << "   Alternative (size): " << greeting.size() << " characters" << std::endl;

    // 4. CHECK IF STRING IS EMPTY
    if (empty.empty()) {
        std::cout << "4. The 'empty' string is indeed empty!" << std::endl;
    }

    if (!greeting.empty()) {
        std::cout << "   The 'greeting' string is NOT empty!" << std::endl;
    }

    // 5. STRING CONCATENATION
    std::string first_name = "John";
    std::string last_name = "Doe";
    std::string full_name = first_name + " " + last_name;
    std::cout << "\n5. Full name: " << full_name << std::endl;

    // 6. APPEND TO STRING (using +=)
    std::string message = "Hello";
    message += " ";
    message += "there!";
    std::cout << "6. Appended message: " << message << std::endl;

    // 7. STRING CAPACITY
    std::cout << "\n7. String capacity info:" << std::endl;
    std::cout << "   Length: " << greeting.length() << std::endl;
    std::cout << "   Capacity: " << greeting.capacity() << std::endl;
    std::cout << "   Max size: " << greeting.max_size() << std::endl;

    // 8. CLEAR A STRING
    std::string temp = "This will be cleared";
    std::cout << "\n8. Before clear: " << temp << std::endl;
    temp.clear();
    std::cout << "   After clear: '" << temp << "' (length: " << temp.length() << ")" << std::endl;

    // 9. WHY USE std::string OVER char[]?
    std::cout << "\n9. Advantages of std::string:" << std::endl;
    std::cout << "   - Automatic memory management" << std::endl;
    std::cout << "   - Dynamic resizing" << std::endl;
    std::cout << "   - Rich set of methods" << std::endl;
    std::cout << "   - Safer and easier to use" << std::endl;

    return 0;
}

/**
 * KEY TAKEAWAYS:
 *
 * 1. Always include <string> header
 * 2. Use std::string (or 'using namespace std;' then just 'string')
 * 3. length() and size() are identical - both return number of characters
 * 4. empty() checks if string has zero length
 * 5. Strings can be concatenated with + or +=
 * 6. Strings automatically manage their memory
 *
 * TRY THIS:
 * - Compile: g++ 01_string_basics.cpp -o basics
 * - Run: ./basics
 */
