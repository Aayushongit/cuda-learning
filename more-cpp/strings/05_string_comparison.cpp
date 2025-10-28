/**
 * 05_string_comparison.cpp
 *
 * String Comparison Operations
 *
 * LEARNING OBJECTIVES:
 * - Compare strings using different methods
 * - Understand lexicographic comparison
 * - Case-sensitive vs case-insensitive comparison
 * - Check for string prefixes and suffixes
 */

#include <iostream>
#include <string>
#include <algorithm>  // For transform

int main() {
    std::cout << "=== STRING COMPARISON ===" << std::endl;

    // 1. EQUALITY COMPARISON (==)
    std::cout << "\n1. EQUALITY COMPARISON (==):" << std::endl;
    std::string str1 = "Hello";
    std::string str2 = "Hello";
    std::string str3 = "hello";

    std::cout << "   \"" << str1 << "\" == \"" << str2 << "\": " << (str1 == str2 ? "true" : "false") << std::endl;
    std::cout << "   \"" << str1 << "\" == \"" << str3 << "\": " << (str1 == str3 ? "true" : "false") << std::endl;

    // 2. INEQUALITY COMPARISON (!=)
    std::cout << "\n2. INEQUALITY COMPARISON (!=):" << std::endl;
    std::cout << "   \"" << str1 << "\" != \"" << str3 << "\": " << (str1 != str3 ? "true" : "false") << std::endl;

    // 3. LEXICOGRAPHIC COMPARISON (<, >, <=, >=)
    std::cout << "\n3. LEXICOGRAPHIC COMPARISON:" << std::endl;
    std::string apple = "apple";
    std::string banana = "banana";

    std::cout << "   \"" << apple << "\" < \"" << banana << "\": " << (apple < banana ? "true" : "false") << std::endl;
    std::cout << "   \"" << banana << "\" > \"" << apple << "\": " << (banana > apple ? "true" : "false") << std::endl;

    // Explanation: 'a' comes before 'b' in ASCII
    std::cout << "   (Alphabetically: apple comes before banana)" << std::endl;

    // 4. COMPARE() METHOD
    std::cout << "\n4. compare() METHOD:" << std::endl;
    std::string word1 = "cat";
    std::string word2 = "dog";
    std::string word3 = "cat";

    int result = word1.compare(word2);
    std::cout << "   \"" << word1 << "\".compare(\"" << word2 << "\"): " << result << std::endl;
    std::cout << "   (negative = word1 < word2)" << std::endl;

    result = word2.compare(word1);
    std::cout << "   \"" << word2 << "\".compare(\"" << word1 << "\"): " << result << std::endl;
    std::cout << "   (positive = word2 > word1)" << std::endl;

    result = word1.compare(word3);
    std::cout << "   \"" << word1 << "\".compare(\"" << word3 << "\"): " << result << std::endl;
    std::cout << "   (zero = equal)" << std::endl;

    // 5. COMPARING SUBSTRINGS
    std::cout << "\n5. COMPARING SUBSTRINGS:" << std::endl;
    std::string text = "Hello World";
    std::string hello = "Hello";

    // compare(pos, len, str)
    result = text.compare(0, 5, hello);
    std::cout << "   First 5 chars of \"" << text << "\" vs \"" << hello << "\": "
              << (result == 0 ? "equal" : "not equal") << std::endl;

    // 6. CASE-SENSITIVE COMPARISON
    std::cout << "\n6. CASE-SENSITIVE COMPARISON:" << std::endl;
    std::string upper = "HELLO";
    std::string lower = "hello";
    std::string mixed = "Hello";

    std::cout << "   \"" << upper << "\" == \"" << lower << "\": " << (upper == lower ? "true" : "false") << std::endl;
    std::cout << "   \"" << upper << "\" == \"" << mixed << "\": " << (upper == mixed ? "true" : "false") << std::endl;
    std::cout << "   (Comparison is case-sensitive by default)" << std::endl;

    // 7. CASE-INSENSITIVE COMPARISON (manual)
    std::cout << "\n7. CASE-INSENSITIVE COMPARISON:" << std::endl;
    std::string s1 = "Hello";
    std::string s2 = "HELLO";

    // Convert both to lowercase for comparison
    std::string s1_lower = s1;
    std::string s2_lower = s2;

    std::transform(s1_lower.begin(), s1_lower.end(), s1_lower.begin(), ::tolower);
    std::transform(s2_lower.begin(), s2_lower.end(), s2_lower.begin(), ::tolower);

    std::cout << "   \"" << s1 << "\" equals \"" << s2 << "\" (ignore case): "
              << (s1_lower == s2_lower ? "true" : "false") << std::endl;

    // 8. CHECKING PREFIX (starts with)
    std::cout << "\n8. CHECKING PREFIX (starts with):" << std::endl;
    std::string filename = "document.txt";
    std::string prefix = "doc";

    // Method 1: Using compare
    bool starts_with = filename.compare(0, prefix.length(), prefix) == 0;
    std::cout << "   \"" << filename << "\" starts with \"" << prefix << "\": "
              << (starts_with ? "true" : "false") << std::endl;

    // Method 2: Using substr (C++20 has starts_with() method)
    starts_with = (filename.substr(0, prefix.length()) == prefix);
    std::cout << "   (Using substr method): " << (starts_with ? "true" : "false") << std::endl;

    // 9. CHECKING SUFFIX (ends with)
    std::cout << "\n9. CHECKING SUFFIX (ends with):" << std::endl;
    std::string file = "photo.jpg";
    std::string suffix = ".jpg";

    bool ends_with = false;
    if (file.length() >= suffix.length()) {
        ends_with = (file.compare(file.length() - suffix.length(), suffix.length(), suffix) == 0);
    }
    std::cout << "   \"" << file << "\" ends with \"" << suffix << "\": "
              << (ends_with ? "true" : "false") << std::endl;

    // 10. EMPTY STRING COMPARISON
    std::cout << "\n10. EMPTY STRING COMPARISON:" << std::endl;
    std::string empty1 = "";
    std::string empty2 = "";
    std::string non_empty = "text";

    std::cout << "   Empty string == Empty string: " << (empty1 == empty2 ? "true" : "false") << std::endl;
    std::cout << "   Empty string == \"text\": " << (empty1 == non_empty ? "true" : "false") << std::endl;

    // 11. COMPARING WITH C-STRING
    std::cout << "\n11. COMPARING WITH C-STRING:" << std::endl;
    std::string cpp_string = "C++";
    const char* c_string = "C++";

    std::cout << "   C++ string vs C-string: " << (cpp_string == c_string ? "equal" : "not equal") << std::endl;

    // 12. LEXICOGRAPHIC ORDER EXAMPLES
    std::cout << "\n12. LEXICOGRAPHIC ORDER EXAMPLES:" << std::endl;
    std::cout << "   \"abc\" < \"abd\": " << ("abc" < std::string("abd") ? "true" : "false") << std::endl;
    std::cout << "   \"abc\" < \"abcd\": " << ("abc" < std::string("abcd") ? "true" : "false") << std::endl;
    std::cout << "   \"ABC\" < \"abc\": " << ("ABC" < std::string("abc") ? "true" : "false") << std::endl;
    std::cout << "   (Uppercase letters come before lowercase in ASCII)" << std::endl;

    return 0;
}

/**
 * COMPARISON OPERATORS SUMMARY:
 *
 * OPERATOR | MEANING                    | EXAMPLE
 * ---------|----------------------------|-----------------------
 * ==       | Equal to                   | str1 == str2
 * !=       | Not equal to               | str1 != str2
 * <        | Less than                  | str1 < str2
 * >        | Greater than               | str1 > str2
 * <=       | Less than or equal to      | str1 <= str2
 * >=       | Greater than or equal to   | str1 >= str2
 *
 * compare() METHOD RETURN VALUES:
 * - Returns 0: strings are equal
 * - Returns < 0: calling string is less than argument
 * - Returns > 0: calling string is greater than argument
 *
 * LEXICOGRAPHIC COMPARISON:
 * - Compares character by character using ASCII values
 * - First different character determines the result
 * - If one string is prefix of another, shorter one is "less"
 * - Examples:
 *   "abc" < "abd"    (c < d)
 *   "abc" < "abcd"   (abc is prefix, shorter is less)
 *   "ABC" < "abc"    (uppercase A=65, lowercase a=97)
 *
 * CASE-INSENSITIVE COMPARISON:
 * C++ strings don't have built-in case-insensitive comparison
 * You need to:
 * 1. Convert both strings to same case (lower or upper)
 * 2. Then compare
 * Use std::transform with ::tolower or ::toupper
 *
 * PREFIX/SUFFIX CHECKING:
 * - C++20 has starts_with() and ends_with() methods
 * - For earlier versions, use compare() or substr()
 *
 * BEST PRACTICES:
 * 1. Use == and != for simple equality checks
 * 2. Use compare() when you need to know the order
 * 3. Be aware of case sensitivity
 * 4. Convert to same case for case-insensitive comparison
 * 5. Check string length before comparing substrings
 *
 * COMPILE AND RUN:
 * g++ 05_string_comparison.cpp -o compare
 * ./compare
 */
