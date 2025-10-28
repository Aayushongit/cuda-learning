/**
 * 04_string_access_iteration.cpp
 *
 * Accessing String Characters and Iterating Through Strings
 *
 * LEARNING OBJECTIVES:
 * - Access individual characters
 * - Iterate through strings using different methods
 * - Understand bounds checking
 * - Modify characters in strings
 */

#include <iostream>
#include <string>

int main() {
    std::cout << "=== STRING ACCESS AND ITERATION ===" << std::endl;

    std::string text = "Hello World";

    // 1. ACCESSING CHARACTERS WITH [] (no bounds checking)
    std::cout << "\n1. ARRAY NOTATION [] ACCESS:" << std::endl;
    std::cout << "   text = \"" << text << "\"" << std::endl;
    std::cout << "   First character: " << text[0] << std::endl;
    std::cout << "   Fifth character: " << text[4] << std::endl;
    std::cout << "   Last character: " << text[text.length() - 1] << std::endl;

    // 2. ACCESSING CHARACTERS WITH at() (with bounds checking)
    std::cout << "\n2. at() METHOD (safer):" << std::endl;
    std::cout << "   text.at(0) = " << text.at(0) << std::endl;
    std::cout << "   text.at(6) = " << text.at(6) << std::endl;

    // Uncomment to see exception:
    // std::cout << text.at(100) << std::endl;  // Throws out_of_range exception

    // 3. FRONT AND BACK
    std::cout << "\n3. FRONT AND BACK METHODS:" << std::endl;
    std::cout << "   First character (front): " << text.front() << std::endl;
    std::cout << "   Last character (back): " << text.back() << std::endl;

    // 4. MODIFYING CHARACTERS
    std::cout << "\n4. MODIFYING CHARACTERS:" << std::endl;
    std::string greeting = "Hello";
    std::cout << "   Original: " << greeting << std::endl;
    greeting[0] = 'h';  // Change 'H' to 'h'
    std::cout << "   Modified: " << greeting << std::endl;

    // 5. ITERATION WITH INDEX-BASED LOOP
    std::cout << "\n5. INDEX-BASED ITERATION:" << std::endl;
    std::cout << "   Characters: ";
    for (size_t i = 0; i < text.length(); i++) {
        std::cout << text[i] << " ";
    }
    std::cout << std::endl;

    // 6. ITERATION WITH RANGE-BASED FOR LOOP (C++11)
    std::cout << "\n6. RANGE-BASED FOR LOOP (modern C++):" << std::endl;
    std::cout << "   Characters: ";
    for (char c : text) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    // 7. ITERATION WITH ITERATORS
    std::cout << "\n7. ITERATOR-BASED ITERATION:" << std::endl;
    std::cout << "   Characters: ";
    for (std::string::iterator it = text.begin(); it != text.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // 8. REVERSE ITERATION
    std::cout << "\n8. REVERSE ITERATION:" << std::endl;
    std::cout << "   Backwards: ";
    for (std::string::reverse_iterator rit = text.rbegin(); rit != text.rend(); ++rit) {
        std::cout << *rit << " ";
    }
    std::cout << std::endl;

    // 9. REVERSE ITERATION (simpler way)
    std::cout << "\n9. REVERSE WITH INDEX:" << std::endl;
    std::cout << "   Backwards: ";
    for (int i = text.length() - 1; i >= 0; i--) {
        std::cout << text[i] << " ";
    }
    std::cout << std::endl;

    // 10. MODIFYING DURING ITERATION
    std::cout << "\n10. MODIFYING DURING ITERATION:" << std::endl;
    std::string word = "hello";
    std::cout << "   Original: " << word << std::endl;

    // Convert to uppercase
    for (size_t i = 0; i < word.length(); i++) {
        if (word[i] >= 'a' && word[i] <= 'z') {
            word[i] = word[i] - 32;  // Convert to uppercase (ASCII math)
        }
    }
    std::cout << "   Uppercase: " << word << std::endl;

    // 11. MODIFYING WITH RANGE-BASED LOOP (need reference)
    std::cout << "\n11. MODIFYING WITH RANGE-BASED LOOP:" << std::endl;
    std::string text2 = "WORLD";
    std::cout << "   Original: " << text2 << std::endl;

    // Using reference (&) to modify characters
    for (char& c : text2) {
        if (c >= 'A' && c <= 'Z') {
            c = c + 32;  // Convert to lowercase
        }
    }
    std::cout << "   Lowercase: " << text2 << std::endl;

    // 12. CONST ITERATION (read-only)
    std::cout << "\n12. CONST ITERATION (read-only):" << std::endl;
    const std::string const_str = "Cannot modify";
    std::cout << "   Reading: ";
    for (const char& c : const_str) {  // const reference
        std::cout << c;
        // c = 'X';  // Error! Cannot modify const
    }
    std::cout << std::endl;

    // 13. COUNTING SPECIFIC CHARACTERS
    std::cout << "\n13. COUNTING SPECIFIC CHARACTERS:" << std::endl;
    std::string sentence = "Hello World";
    int count_l = 0;
    for (char c : sentence) {
        if (c == 'l') {
            count_l++;
        }
    }
    std::cout << "   Number of 'l' in \"" << sentence << "\": " << count_l << std::endl;

    // 14. PRINTING WITH INDICES
    std::cout << "\n14. CHARACTERS WITH POSITIONS:" << std::endl;
    std::string example = "C++";
    for (size_t i = 0; i < example.length(); i++) {
        std::cout << "   Index " << i << ": " << example[i] << std::endl;
    }

    return 0;
}

/**
 * ACCESS METHODS COMPARISON:
 *
 * METHOD          | BOUNDS CHECK | SPEED    | THROWS EXCEPTION | USE CASE
 * ----------------|--------------|----------|------------------|---------------------------
 * str[i]          | NO           | Fastest  | NO (undefined)   | When index is guaranteed valid
 * str.at(i)       | YES          | Slower   | YES (safe)       | When safety is important
 * str.front()     | NO           | Fast     | NO               | First character
 * str.back()      | NO           | Fast     | NO               | Last character
 *
 * ITERATION METHODS:
 *
 * 1. INDEX-BASED (for loop with i):
 *    - Full control over index
 *    - Can iterate backwards easily
 *    - Can skip elements
 *
 * 2. RANGE-BASED (for char c : str):
 *    - Clean and readable
 *    - Modern C++ style
 *    - Use & for modification: for (char& c : str)
 *
 * 3. ITERATORS:
 *    - More flexible
 *    - Works with algorithms
 *    - begin()/end() for forward
 *    - rbegin()/rend() for reverse
 *
 * KEY POINTS:
 * - Use at() when you need safety (catches invalid indices)
 * - Use [] when you're sure the index is valid (faster)
 * - Use range-based for loop for cleaner code
 * - Use & (reference) in range-based loop to modify characters
 * - Remember: string indices start at 0
 * - Last character is at index (length - 1)
 *
 * COMPILE AND RUN:
 * g++ 04_string_access_iteration.cpp -o access
 * ./access
 */
