/**
 * 07_string_searching.cpp
 *
 * String Searching and Finding Operations
 *
 * LEARNING OBJECTIVES:
 * - Find substrings and characters
 * - Search from different positions
 * - Reverse searching
 * - Find first/last occurrences
 * - Check if substring exists
 */

#include <iostream>
#include <string>

int main() {
    std::cout << "=== STRING SEARCHING AND FINDING ===" << std::endl;

    // 1. BASIC FIND - Search for substring
    std::cout << "\n1. BASIC FIND:" << std::endl;
    std::string text = "Hello World, Welcome to C++ World";
    std::string search = "World";

    size_t pos = text.find(search);
    if (pos != std::string::npos) {
        std::cout << "   \"" << search << "\" found at position: " << pos << std::endl;
    } else {
        std::cout << "   \"" << search << "\" not found" << std::endl;
    }

    // 2. FINDING CHARACTERS
    std::cout << "\n2. FINDING SINGLE CHARACTER:" << std::endl;
    pos = text.find('W');
    std::cout << "   'W' found at position: " << pos << std::endl;

    // 3. FIND STARTING FROM A POSITION
    std::cout << "\n3. FIND FROM SPECIFIC POSITION:" << std::endl;
    std::cout << "   Text: \"" << text << "\"" << std::endl;

    size_t first_world = text.find("World");
    std::cout << "   First \"World\" at: " << first_world << std::endl;

    size_t second_world = text.find("World", first_world + 1);
    if (second_world != std::string::npos) {
        std::cout << "   Second \"World\" at: " << second_world << std::endl;
    }

    // 4. FINDING ALL OCCURRENCES
    std::cout << "\n4. FINDING ALL OCCURRENCES:" << std::endl;
    std::string haystack = "The cat in the hat sat on the mat";
    std::string needle = "at";

    std::cout << "   Searching for \"" << needle << "\" in:" << std::endl;
    std::cout << "   \"" << haystack << "\"" << std::endl;
    std::cout << "   Positions: ";

    pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        std::cout << pos << " ";
        pos++;  // Move past this occurrence
    }
    std::cout << std::endl;

    // 5. RFIND - Search backwards (from end)
    std::cout << "\n5. RFIND (reverse find):" << std::endl;
    std::string sentence = "First World, Second World, Third World";
    std::cout << "   Text: \"" << sentence << "\"" << std::endl;

    size_t last_pos = sentence.rfind("World");
    std::cout << "   Last \"World\" at position: " << last_pos << std::endl;

    size_t first_pos = sentence.find("World");
    std::cout << "   First \"World\" at position: " << first_pos << std::endl;

    // 6. FIND_FIRST_OF - Find any character from a set
    std::cout << "\n6. FIND_FIRST_OF (any char from set):" << std::endl;
    std::string data = "Hello123World";
    std::string digits = "0123456789";

    pos = data.find_first_of(digits);
    if (pos != std::string::npos) {
        std::cout << "   First digit in \"" << data << "\" found at: " << pos << std::endl;
        std::cout << "   The digit is: " << data[pos] << std::endl;
    }

    // 7. FIND_LAST_OF - Find last occurrence of any character from set
    std::cout << "\n7. FIND_LAST_OF:" << std::endl;
    std::string path = "documents/work/report.txt";
    std::string separators = "/\\";

    pos = path.find_last_of(separators);
    if (pos != std::string::npos) {
        std::cout << "   Path: \"" << path << "\"" << std::endl;
        std::cout << "   Last separator at: " << pos << std::endl;
        std::cout << "   Filename: " << path.substr(pos + 1) << std::endl;
    }

    // 8. FIND_FIRST_NOT_OF - Find first character NOT in set
    std::cout << "\n8. FIND_FIRST_NOT_OF:" << std::endl;
    std::string padded = "   Hello";
    std::string whitespace = " \t\n";

    pos = padded.find_first_not_of(whitespace);
    std::cout << "   Original: \"" << padded << "\"" << std::endl;
    std::cout << "   First non-space at: " << pos << std::endl;
    std::cout << "   Trimmed: \"" << padded.substr(pos) << "\"" << std::endl;

    // 9. FIND_LAST_NOT_OF - Find last character NOT in set
    std::cout << "\n9. FIND_LAST_NOT_OF:" << std::endl;
    std::string trailing = "Hello   ";

    pos = trailing.find_last_not_of(whitespace);
    std::cout << "   Original: \"" << trailing << "\"" << std::endl;
    std::cout << "   Last non-space at: " << pos << std::endl;
    std::cout << "   Trimmed: \"" << trailing.substr(0, pos + 1) << "\"" << std::endl;

    // 10. CHECKING IF SUBSTRING EXISTS
    std::cout << "\n10. CHECKING SUBSTRING EXISTENCE:" << std::endl;
    std::string document = "C++ is a powerful programming language";

    if (document.find("powerful") != std::string::npos) {
        std::cout << "   \"powerful\" exists in the document" << std::endl;
    }

    if (document.find("Java") == std::string::npos) {
        std::cout << "   \"Java\" does NOT exist in the document" << std::endl;
    }

    // 11. COUNTING OCCURRENCES
    std::cout << "\n11. COUNTING OCCURRENCES:" << std::endl;
    std::string sample = "banana";
    char target = 'a';
    int count = 0;

    pos = 0;
    while ((pos = sample.find(target, pos)) != std::string::npos) {
        count++;
        pos++;
    }
    std::cout << "   \"" << target << "\" appears " << count << " times in \"" << sample << "\"" << std::endl;

    // 12. CASE-INSENSITIVE SEARCH (manual approach)
    std::cout << "\n12. CASE-INSENSITIVE SEARCH:" << std::endl;
    std::string original = "Hello WORLD";
    std::string search_term = "world";

    // Convert both to lowercase
    std::string lower_original = original;
    for (char& c : lower_original) {
        c = std::tolower(c);
    }

    pos = lower_original.find(search_term);
    if (pos != std::string::npos) {
        std::cout << "   \"" << search_term << "\" found (case-insensitive) at: " << pos << std::endl;
    }

    // 13. PRACTICAL EXAMPLE - Email validation
    std::cout << "\n13. PRACTICAL EXAMPLE (Email check):" << std::endl;
    std::string email1 = "user@example.com";
    std::string email2 = "invalid.email.com";

    auto validate_email = [](const std::string& email) {
        size_t at_pos = email.find('@');
        size_t dot_pos = email.find('.', at_pos);

        return (at_pos != std::string::npos &&
                dot_pos != std::string::npos &&
                at_pos > 0 &&
                dot_pos > at_pos + 1 &&
                dot_pos < email.length() - 1);
    };

    std::cout << "   " << email1 << ": " << (validate_email(email1) ? "Valid" : "Invalid") << std::endl;
    std::cout << "   " << email2 << ": " << (validate_email(email2) ? "Valid" : "Invalid") << std::endl;

    return 0;
}

/**
 * STRING SEARCH METHODS SUMMARY:
 *
 * METHOD               | PURPOSE                           | RETURNS
 * ---------------------|-----------------------------------|---------------------------
 * find()               | Find first occurrence             | Position or npos
 * rfind()              | Find last occurrence              | Position or npos
 * find_first_of()      | Find any char from set            | Position or npos
 * find_last_of()       | Find last occurrence of any char  | Position or npos
 * find_first_not_of()  | Find first char NOT in set        | Position or npos
 * find_last_not_of()   | Find last char NOT in set         | Position or npos
 *
 * IMPORTANT CONSTANT:
 * string::npos
 * - Represents "not found"
 * - Value is usually -1 (maximum value of size_t)
 * - Always check: if (pos != string::npos)
 *
 * FIND VARIANTS:
 * 1. find(str):           Search for substring from beginning
 * 2. find(str, pos):      Search for substring starting at pos
 * 3. find(char):          Search for single character
 * 4. find(char, pos):     Search for character starting at pos
 *
 * COMMON PATTERNS:
 *
 * 1. Find all occurrences:
 *    pos = 0;
 *    while ((pos = str.find(substr, pos)) != npos) {
 *        // Found at pos
 *        pos++;
 *    }
 *
 * 2. Check if substring exists:
 *    if (str.find(substr) != npos) {
 *        // Found
 *    }
 *
 * 3. Extract filename from path:
 *    pos = path.find_last_of("/\\");
 *    filename = path.substr(pos + 1);
 *
 * 4. Trim whitespace:
 *    start = str.find_first_not_of(" \t\n");
 *    end = str.find_last_not_of(" \t\n");
 *    trimmed = str.substr(start, end - start + 1);
 *
 * PERFORMANCE NOTES:
 * - find() is generally O(n*m) where n=string length, m=pattern length
 * - For multiple searches, consider using algorithms from <algorithm>
 * - Case-insensitive search requires converting both strings (overhead)
 *
 * TIPS:
 * 1. Always check for npos before using the position
 * 2. Use find_first_of() for finding any of several characters
 * 3. Use rfind() to search from the end
 * 4. Remember to increment position when finding all occurrences
 *
 * COMPILE AND RUN:
 * g++ 07_string_searching.cpp -o search
 * ./search
 */
