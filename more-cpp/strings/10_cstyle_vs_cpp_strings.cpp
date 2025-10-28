/**
 * 10_cstyle_vs_cpp_strings.cpp
 *
 * C-Style Strings vs C++ Strings Comparison
 *
 * LEARNING OBJECTIVES:
 * - Understand C-style strings (char arrays)
 * - Compare with C++ std::string
 * - Learn conversion between the two
 * - Know when to use each type
 * - Understand null termination
 */

#include <iostream>
#include <string>
#include <cstring>  // For C-string functions (strlen, strcpy, etc.)

int main() {
    std::cout << "=== C-STYLE vs C++ STRINGS ===" << std::endl;

    // 1. C-STYLE STRING DECLARATION
    std::cout << "\n1. C-STYLE STRING:" << std::endl;
    char c_str1[] = "Hello";  // Automatically null-terminated
    char c_str2[20] = "World"; // Fixed size array
    const char* c_str3 = "Literal";  // String literal (read-only)

    std::cout << "   c_str1: " << c_str1 << std::endl;
    std::cout << "   c_str2: " << c_str2 << std::endl;
    std::cout << "   c_str3: " << c_str3 << std::endl;

    // 2. C++ STRING DECLARATION
    std::cout << "\n2. C++ STRING:" << std::endl;
    std::string cpp_str1 = "Hello";
    std::string cpp_str2 = "World";

    std::cout << "   cpp_str1: " << cpp_str1 << std::endl;
    std::cout << "   cpp_str2: " << cpp_str2 << std::endl;

    // 3. NULL TERMINATION
    std::cout << "\n3. NULL TERMINATION:" << std::endl;
    char manual[6] = {'H', 'e', 'l', 'l', 'o', '\0'};  // Must add \0
    std::cout << "   Manual C-string: " << manual << std::endl;
    std::cout << "   Size of array: " << sizeof(manual) << " bytes" << std::endl;
    std::cout << "   Actual length: " << strlen(manual) << " characters" << std::endl;

    // 4. SIZE AND LENGTH
    std::cout << "\n4. SIZE AND LENGTH COMPARISON:" << std::endl;
    char c_array[50] = "Test";
    std::string cpp_string = "Test";

    std::cout << "   C-style:" << std::endl;
    std::cout << "   sizeof(array): " << sizeof(c_array) << " bytes (array size)" << std::endl;
    std::cout << "   strlen(): " << strlen(c_array) << " characters" << std::endl;

    std::cout << "   C++ style:" << std::endl;
    std::cout << "   length(): " << cpp_string.length() << " characters" << std::endl;
    std::cout << "   size(): " << cpp_string.size() << " characters" << std::endl;

    // 5. CONCATENATION COMPARISON
    std::cout << "\n5. CONCATENATION:" << std::endl;

    // C-style (manual, error-prone)
    char c_result[50] = "Hello ";
    strcat(c_result, "World");  // Dangerous! Must ensure buffer is large enough
    std::cout << "   C-style result: " << c_result << std::endl;

    // C++ style (easy and safe)
    std::string cpp_result = "Hello " + std::string("World");
    std::cout << "   C++ result: " << cpp_result << std::endl;

    // 6. COPYING STRINGS
    std::cout << "\n6. COPYING STRINGS:" << std::endl;

    // C-style
    char c_source[] = "Copy me";
    char c_dest[20];
    strcpy(c_dest, c_source);  // Unsafe if dest too small
    std::cout << "   C-style copied: " << c_dest << std::endl;

    // C++ style
    std::string cpp_source = "Copy me";
    std::string cpp_dest = cpp_source;  // Safe, automatic memory management
    std::cout << "   C++ copied: " << cpp_dest << std::endl;

    // 7. COMPARISON
    std::cout << "\n7. STRING COMPARISON:" << std::endl;

    // C-style
    char c_a[] = "abc";
    char c_b[] = "abc";
    int c_compare = strcmp(c_a, c_b);  // Returns 0 if equal
    std::cout << "   C-style compare result: " << c_compare << std::endl;
    std::cout << "   (0 = equal, <0 = first<second, >0 = first>second)" << std::endl;

    // C++ style
    std::string cpp_a = "abc";
    std::string cpp_b = "abc";
    bool cpp_equal = (cpp_a == cpp_b);
    std::cout << "   C++ comparison: " << (cpp_equal ? "equal" : "not equal") << std::endl;

    // 8. CONVERTING C++ STRING TO C-STRING
    std::cout << "\n8. C++ STRING TO C-STRING:" << std::endl;
    std::string cpp_original = "Convert me";
    const char* c_ptr = cpp_original.c_str();  // Get C-string pointer
    std::cout << "   C++ string: " << cpp_original << std::endl;
    std::cout << "   C-string pointer: " << c_ptr << std::endl;

    // Alternative: data() method
    const char* c_ptr2 = cpp_original.data();
    std::cout << "   Using data(): " << c_ptr2 << std::endl;

    // 9. CONVERTING C-STRING TO C++ STRING
    std::cout << "\n9. C-STRING TO C++ STRING:" << std::endl;
    const char* c_input = "Convert to C++";
    std::string cpp_converted(c_input);  // Constructor
    std::string cpp_assigned = c_input;  // Assignment
    std::cout << "   C-string: " << c_input << std::endl;
    std::cout << "   C++ string: " << cpp_converted << std::endl;

    // 10. DYNAMIC SIZING
    std::cout << "\n10. DYNAMIC SIZING:" << std::endl;

    // C-style: Fixed size or manual memory management
    char fixed[10] = "Short";
    std::cout << "   C-style (fixed): " << fixed << std::endl;
    // fixed = "This is too long";  // ERROR! Cannot assign, would overflow

    // C++ style: Automatic resizing
    std::string dynamic = "Short";
    std::cout << "   C++ string: " << dynamic << std::endl;
    dynamic = "This can be as long as needed without problems!";
    std::cout << "   Resized: " << dynamic << std::endl;

    // 11. SAFETY COMPARISON
    std::cout << "\n11. SAFETY ISSUES:" << std::endl;
    std::cout << "   C-style strings can cause:" << std::endl;
    std::cout << "   - Buffer overflows (security risk)" << std::endl;
    std::cout << "   - Memory leaks (if dynamically allocated)" << std::endl;
    std::cout << "   - Null pointer dereferences" << std::endl;
    std::cout << "\n   C++ strings are:" << std::endl;
    std::cout << "   - Automatically sized" << std::endl;
    std::cout << "   - Memory-safe" << std::endl;
    std::cout << "   - Exception-safe" << std::endl;

    // 12. COMMON C-STRING FUNCTIONS
    std::cout << "\n12. C-STRING FUNCTIONS vs C++ METHODS:" << std::endl;
    std::cout << "   strlen()   -> string.length()" << std::endl;
    std::cout << "   strcpy()   -> string assignment (=)" << std::endl;
    std::cout << "   strcat()   -> string.append() or +=" << std::endl;
    std::cout << "   strcmp()   -> == operator" << std::endl;
    std::cout << "   strchr()   -> string.find()" << std::endl;
    std::cout << "   strstr()   -> string.find()" << std::endl;

    // 13. WHEN TO USE EACH
    std::cout << "\n13. WHEN TO USE EACH TYPE:" << std::endl;
    std::cout << "   Use C-style strings when:" << std::endl;
    std::cout << "   - Interfacing with C libraries" << std::endl;
    std::cout << "   - Working with legacy code" << std::endl;
    std::cout << "   - Extreme performance requirements" << std::endl;
    std::cout << "\n   Use C++ strings when:" << std::endl;
    std::cout << "   - Writing new C++ code (default choice)" << std::endl;
    std::cout << "   - Need dynamic sizing" << std::endl;
    std::cout << "   - Want safety and ease of use" << std::endl;
    std::cout << "   - Using STL containers and algorithms" << std::endl;

    // 14. PRACTICAL EXAMPLE - Function parameters
    std::cout << "\n14. FUNCTION PARAMETERS:" << std::endl;

    // Function accepting C-string
    auto print_c_string = [](const char* str) {
        std::cout << "   C-string param: " << str << std::endl;
    };

    // Function accepting C++ string
    auto print_cpp_string = [](const std::string& str) {
        std::cout << "   C++ string param: " << str << std::endl;
    };

    std::string my_string = "Test";
    print_c_string(my_string.c_str());  // Need to convert
    print_cpp_string(my_string);         // Direct pass

    // 15. MIXED USAGE
    std::cout << "\n15. MIXED USAGE EXAMPLE:" << std::endl;
    std::string filename = "data.txt";

    // Many C functions require C-strings
    // FILE* file = fopen(filename.c_str(), "r");  // Need c_str()

    std::cout << "   C++ string: " << filename << std::endl;
    std::cout << "   Converted to C-string: " << filename.c_str() << std::endl;
    std::cout << "   (Required for many C library functions)" << std::endl;

    return 0;
}

/**
 * C-STYLE vs C++ STRINGS COMPARISON:
 *
 * FEATURE              | C-STYLE (char*)        | C++ (std::string)
 * ---------------------|------------------------|-------------------------
 * Declaration          | char str[50];          | string str;
 * Memory management    | Manual                 | Automatic
 * Dynamic sizing       | No (or manual)         | Yes (automatic)
 * Null termination     | Required (\0)          | Handled internally
 * Concatenation        | strcat() (unsafe)      | + or += (safe)
 * Comparison           | strcmp()               | == operator
 * Length               | strlen()               | .length() or .size()
 * Copy                 | strcpy() (unsafe)      | = operator
 * Safety               | Error-prone            | Safe
 * Performance          | Slightly faster        | Very close
 *
 * KEY DIFFERENCES:
 *
 * 1. MEMORY MANAGEMENT:
 *    C-style: char str[100];  // Fixed, or new/delete
 *    C++:     string str;     // Automatic
 *
 * 2. NULL TERMINATION:
 *    C-style: Must ensure \0 at end
 *    C++:     Handled automatically
 *
 * 3. SAFETY:
 *    C-style: strcpy(dest, src);  // Can overflow!
 *    C++:     dest = src;          // Always safe
 *
 * 4. CONCATENATION:
 *    C-style: strcat(dest, src);   // Risk of overflow
 *    C++:     dest += src;         // Safe, auto-resizes
 *
 * CONVERSION BETWEEN TYPES:
 *
 * C++ to C-style:
 * - string.c_str()     Returns const char* (preferred)
 * - string.data()      Returns const char* (C++11)
 *
 * C-style to C++:
 * - string str(c_str)  Constructor
 * - string str = c_str Assignment
 *
 * COMMON C-STRING FUNCTIONS (from <cstring>):
 * - strlen(str)        Length
 * - strcpy(dest, src)  Copy
 * - strcat(dest, src)  Concatenate
 * - strcmp(s1, s2)     Compare
 * - strchr(str, ch)    Find character
 * - strstr(s1, s2)     Find substring
 *
 * BEST PRACTICES:
 * 1. Prefer std::string in new C++ code
 * 2. Use C-strings only when necessary (C APIs)
 * 3. Never use strcpy/strcat - use strncpy/strncat if needed
 * 4. Convert to C-string only when calling C functions
 * 5. Be aware that c_str() pointer can become invalid if string changes
 *
 * SECURITY NOTE:
 * Many C-string functions (strcpy, strcat, gets, etc.) are considered
 * unsafe and can lead to buffer overflows. Use C++ strings or safer
 * alternatives (strncpy, strncat, fgets, etc.)
 *
 * COMPILE AND RUN:
 * g++ 10_cstyle_vs_cpp_strings.cpp -o cstyle
 * ./cstyle
 */
