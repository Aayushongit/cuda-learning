/**
 * 03_string_input_output.cpp
 *
 * String Input and Output Operations
 *
 * LEARNING OBJECTIVES:
 * - Read strings from user input
 * - Output strings to console
 * - Handle whitespace in input
 * - Read entire lines vs single words
 */

#include <iostream>
#include <string>

int main() {
    std::cout << "=== STRING INPUT/OUTPUT ===" << std::endl;

    // 1. BASIC OUTPUT
    std::cout << "\n1. BASIC OUTPUT:" << std::endl;
    std::string message = "Hello, C++!";
    std::cout << "   Message: " << message << std::endl;

    // 2. CONCATENATED OUTPUT
    std::string first = "John";
    std::string last = "Doe";
    std::cout << "\n2. CONCATENATED OUTPUT:" << std::endl;
    std::cout << "   Name: " << first << " " << last << std::endl;

    // 3. BASIC INPUT (reads until whitespace)
    std::cout << "\n3. BASIC INPUT (single word):" << std::endl;
    std::cout << "   Enter your first name: ";
    std::string name;
    std::cin >> name;  // Stops at first whitespace
    std::cout << "   Hello, " << name << "!" << std::endl;

    // IMPORTANT: Clear the input buffer before using getline
    std::cin.ignore(10000, '\n');

    // 4. READING A FULL LINE (including spaces)
    std::cout << "\n4. READING FULL LINE:" << std::endl;
    std::cout << "   Enter your full name: ";
    std::string full_name;
    std::getline(std::cin, full_name);  // Reads entire line including spaces
    std::cout << "   Full name: " << full_name << std::endl;

    // 5. READING MULTIPLE WORDS
    std::cout << "\n5. READING MULTIPLE WORDS (separately):" << std::endl;
    std::cout << "   Enter two words separated by space: ";
    std::string word1, word2;
    std::cin >> word1 >> word2;
    std::cout << "   Word 1: " << word1 << std::endl;
    std::cout << "   Word 2: " << word2 << std::endl;

    // Clear buffer again
    std::cin.ignore(10000, '\n');

    // 6. READING WITH CUSTOM DELIMITER
    std::cout << "\n6. READING WITH CUSTOM DELIMITER:" << std::endl;
    std::cout << "   Enter text ending with '#': ";
    std::string custom_input;
    std::getline(std::cin, custom_input, '#');  // Read until '#'
    std::cout << "   You entered: " << custom_input << std::endl;

    // Clear remaining input
    std::cin.ignore(10000, '\n');

    // 7. READING MULTIPLE LINES
    std::cout << "\n7. READING MULTIPLE LINES:" << std::endl;
    std::cout << "   Enter line 1: ";
    std::string line1;
    std::getline(std::cin, line1);

    std::cout << "   Enter line 2: ";
    std::string line2;
    std::getline(std::cin, line2);

    std::cout << "   Line 1: " << line1 << std::endl;
    std::cout << "   Line 2: " << line2 << std::endl;

    // 8. FORMATTED OUTPUT
    std::cout << "\n8. FORMATTED OUTPUT:" << std::endl;
    std::string product = "Laptop";
    int price = 999;
    std::cout << "   Product: " << product << std::endl;
    std::cout << "   Price: $" << price << std::endl;

    // 9. OUTPUT WITH MANIPULATORS
    std::cout << "\n9. OUTPUT WITH ESCAPE SEQUENCES:" << std::endl;
    std::string quote = "\"To be or not to be\"";
    std::cout << "   Quote: " << quote << std::endl;
    std::cout << "   Tab-separated:\tValue1\tValue2\tValue3" << std::endl;
    std::cout << "   Newline\nNew line here" << std::endl;

    // 10. CHECKING INPUT SUCCESS
    std::cout << "\n10. INPUT VALIDATION:" << std::endl;
    std::cout << "   Enter a word: ";
    std::string validated_input;
    if (std::cin >> validated_input) {
        std::cout << "   Successfully read: " << validated_input << std::endl;
    } else {
        std::cout << "   Input failed!" << std::endl;
    }

    return 0;
}

/**
 * INPUT/OUTPUT COMPARISON:
 *
 * METHOD              | SYNTAX                        | STOPS AT        | USE CASE
 * --------------------|-------------------------------|-----------------|------------------------
 * cin >>              | cin >> str;                   | Whitespace      | Single word input
 * getline()           | getline(cin, str);            | Newline         | Full line with spaces
 * getline(delimiter)  | getline(cin, str, delim);     | Custom char     | Custom parsing
 *
 * COMMON PITFALLS:
 *
 * 1. MIXING cin >> and getline():
 *    Problem: cin >> leaves newline in buffer, which getline() immediately reads
 *    Solution: Use cin.ignore() after cin >>
 *
 *    Example:
 *    cin >> age;              // Reads number, leaves '\n'
 *    cin.ignore();            // Remove the '\n'
 *    getline(cin, name);      // Now works correctly
 *
 * 2. WHITESPACE IN INPUT:
 *    cin >> reads: "Hello"    (stops at space)
 *    getline() reads: "Hello World"  (reads entire line)
 *
 * 3. EMPTY INPUT:
 *    If user just presses Enter, getline() stores an empty string
 *    Check with: if (str.empty()) { ... }
 *
 * KEY POINTS:
 * - Use cin >> for single words (no spaces)
 * - Use getline() for full lines (with spaces)
 * - Always clear input buffer when mixing methods
 * - Validate input to ensure it was successful
 *
 * COMPILE AND RUN:
 * g++ 03_string_input_output.cpp -o io
 * ./io
 *
 * TRY THESE INPUTS:
 * - Single words
 * - Full sentences with spaces
 * - Empty input (just press Enter)
 * - Text with special characters
 */
