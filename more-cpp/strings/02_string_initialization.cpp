/**
 * 02_string_initialization.cpp
 *
 * Different Ways to Initialize and Declare Strings
 *
 * LEARNING OBJECTIVES:
 * - Understand various string initialization methods
 * - Learn constructor variations
 * - Know when to use each method
 */

#include <iostream>
#include <string>

int main() {
    std::cout << "=== STRING INITIALIZATION METHODS ===" << std::endl;

    // 1. DEFAULT INITIALIZATION (empty string)
    std::string str1;
    std::cout << "\n1. Default initialization:" << std::endl;
    std::cout << "   str1: '" << str1 << "' (length: " << str1.length() << ")" << std::endl;

    // 2. DIRECT INITIALIZATION with string literal
    std::string str2 = "Hello World";
    std::cout << "\n2. Direct initialization:" << std::endl;
    std::cout << "   str2: " << str2 << std::endl;

    // 3. UNIFORM INITIALIZATION (C++11)
    std::string str3{"C++ Strings"};
    std::cout << "\n3. Uniform initialization:" << std::endl;
    std::cout << "   str3: " << str3 << std::endl;

    // 4. COPY INITIALIZATION
    std::string str4 = str2;
    std::cout << "\n4. Copy initialization:" << std::endl;
    std::cout << "   str4 (copy of str2): " << str4 << std::endl;

    // 5. INITIALIZATION WITH REPEATED CHARACTERS
    std::string str5(10, '*');  // 10 asterisks
    std::cout << "\n5. Repeated character initialization:" << std::endl;
    std::cout << "   str5: " << str5 << std::endl;

    // 6. INITIALIZATION FROM SUBSTRING
    std::string original = "Hello World";
    std::string str6(original, 0, 5);  // From position 0, take 5 characters
    std::cout << "\n6. Substring initialization:" << std::endl;
    std::cout << "   original: " << original << std::endl;
    std::cout << "   str6 (first 5 chars): " << str6 << std::endl;

    // 7. INITIALIZATION FROM C-STRING (char array)
    const char* c_string = "C-style string";
    std::string str7(c_string);
    std::cout << "\n7. From C-string:" << std::endl;
    std::cout << "   str7: " << str7 << std::endl;

    // 8. INITIALIZATION WITH PART OF C-STRING
    const char* long_cstring = "ABCDEFGHIJ";
    std::string str8(long_cstring, 5);  // First 5 characters only
    std::cout << "\n8. Partial C-string initialization:" << std::endl;
    std::cout << "   str8 (first 5 chars): " << str8 << std::endl;

    // 9. INITIALIZATION FROM ITERATORS
    std::string base = "Programming";
    std::string str9(base.begin(), base.begin() + 7);  // First 7 characters
    std::cout << "\n9. Iterator-based initialization:" << std::endl;
    std::cout << "   str9: " << str9 << std::endl;

    // 10. MOVE INITIALIZATION (C++11) - efficient transfer
    std::string temp = "Temporary string";
    std::string str10 = std::move(temp);
    std::cout << "\n10. Move initialization:" << std::endl;
    std::cout << "   str10: " << str10 << std::endl;
    std::cout << "   temp after move: '" << temp << "' (may be empty)" << std::endl;

    // 11. ASSIGNMENT vs INITIALIZATION
    std::cout << "\n11. Assignment vs Initialization:" << std::endl;
    std::string str11 = "initialization";  // Initialization
    std::cout << "   str11 (initialized): " << str11 << std::endl;
    str11 = "assignment";  // Assignment
    std::cout << "   str11 (assigned): " << str11 << std::endl;

    // 12. RESERVE CAPACITY (optimization)
    std::string str12;
    str12.reserve(100);  // Reserve space for 100 characters
    std::cout << "\n12. Reserved capacity:" << std::endl;
    std::cout << "   str12 length: " << str12.length() << std::endl;
    std::cout << "   str12 capacity: " << str12.capacity() << std::endl;

    // 13. MULTIPLE STRINGS IN ONE LINE
    std::string name = "Alice", city = "Paris", country = "France";
    std::cout << "\n13. Multiple declarations:" << std::endl;
    std::cout << "   " << name << " lives in " << city << ", " << country << std::endl;

    return 0;
}

/**
 * INITIALIZATION SUMMARY:
 *
 * METHOD                          | SYNTAX                        | USE CASE
 * --------------------------------|-------------------------------|---------------------------
 * Default                         | string s;                     | Empty string, fill later
 * Direct                          | string s = "text";            | Most common
 * Uniform (C++11)                 | string s{"text"};             | Modern C++ style
 * Copy                            | string s2 = s1;               | Duplicate existing string
 * Repeated character              | string s(n, 'c');             | Create pattern
 * Substring                       | string s(str, pos, len);      | Extract part of string
 * From C-string                   | string s(cstr);               | Convert C-style to C++
 * From iterators                  | string s(begin, end);         | Range-based creation
 * Move (C++11)                    | string s = move(tmp);         | Efficient transfer
 *
 * BEST PRACTICES:
 * 1. Use direct initialization (=) for readability
 * 2. Use uniform initialization ({}) for consistency in modern C++
 * 3. Use reserve() when you know the approximate final size
 * 4. Use move semantics for large strings to avoid copying
 *
 * COMPILE AND RUN:
 * g++ 02_string_initialization.cpp -o init
 * ./init
 */
