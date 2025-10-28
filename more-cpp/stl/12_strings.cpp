/**
 * 12_strings.cpp
 *
 * STRING AND STRING_VIEW
 * - std::string operations
 * - std::string_view (C++17) - Non-owning string reference
 * - String manipulation
 * - Searching, replacing, formatting
 */

#include <iostream>
#include <string>
#include <string_view>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <vector>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== STRINGS AND STRING_VIEW ===\n";

    // ========== STRING BASICS ==========
    separator("STRING BASICS");

    // 1. String Creation
    std::cout << "\n1. STRING CREATION:\n";
    std::string s1;                              // Empty
    std::string s2 = "Hello";                    // From literal
    std::string s3("World");                     // Constructor
    std::string s4(5, 'A');                      // Repeat character
    std::string s5(s2);                          // Copy
    std::string s6 = s2 + " " + s3;             // Concatenation

    std::cout << "s2: " << s2 << "\n";
    std::cout << "s4: " << s4 << "\n";
    std::cout << "s6: " << s6 << "\n";

    // 2. String Properties
    std::cout << "\n2. STRING PROPERTIES:\n";
    std::string str = "Hello, World!";

    std::cout << "length(): " << str.length() << "\n";
    std::cout << "size(): " << str.size() << "\n";
    std::cout << "capacity(): " << str.capacity() << "\n";
    std::cout << "max_size(): " << str.max_size() << "\n";
    std::cout << "empty(): " << (str.empty() ? "true" : "false") << "\n";

    // 3. Accessing Characters
    std::cout << "\n3. ACCESSING CHARACTERS:\n";
    std::string text = "Programming";

    std::cout << "text[0]: " << text[0] << "\n";
    std::cout << "text.at(1): " << text.at(1) << "\n";
    std::cout << "text.front(): " << text.front() << "\n";
    std::cout << "text.back(): " << text.back() << "\n";

    // Access raw data
    const char* c_str = text.c_str();
    std::cout << "c_str(): " << c_str << "\n";

    // 4. Modifying Strings
    std::cout << "\n4. MODIFYING STRINGS:\n";
    std::string modify = "Hello";

    modify += " World";                    // Append
    std::cout << "After +=: " << modify << "\n";

    modify.append("!!!");
    std::cout << "After append: " << modify << "\n";

    modify.push_back('?');
    std::cout << "After push_back: " << modify << "\n";

    modify.pop_back();
    std::cout << "After pop_back: " << modify << "\n";

    modify.insert(5, " Beautiful");
    std::cout << "After insert: " << modify << "\n";

    modify.erase(5, 10);  // Erase " Beautiful"
    std::cout << "After erase: " << modify << "\n";

    modify.replace(0, 5, "Hi");
    std::cout << "After replace: " << modify << "\n";

    modify.clear();
    std::cout << "After clear, size: " << modify.size() << "\n";

    // 5. Substring
    std::cout << "\n5. SUBSTRING:\n";
    std::string full = "Hello World Programming";

    std::string sub1 = full.substr(0, 5);        // "Hello"
    std::string sub2 = full.substr(6, 5);        // "World"
    std::string sub3 = full.substr(12);          // "Programming"

    std::cout << "substr(0, 5): " << sub1 << "\n";
    std::cout << "substr(6, 5): " << sub2 << "\n";
    std::cout << "substr(12): " << sub3 << "\n";

    // 6. Searching
    std::cout << "\n6. SEARCHING:\n";
    std::string search_str = "Hello World, World is beautiful";

    size_t pos1 = search_str.find("World");
    std::cout << "find('World'): " << pos1 << "\n";

    size_t pos2 = search_str.find("World", pos1 + 1);  // Find next
    std::cout << "find('World', from " << (pos1 + 1) << "): " << pos2 << "\n";

    size_t pos3 = search_str.rfind("World");  // Reverse find
    std::cout << "rfind('World'): " << pos3 << "\n";

    size_t pos4 = search_str.find_first_of("aeiou");
    std::cout << "find_first_of('aeiou'): " << pos4 << " ('" << search_str[pos4] << "')\n";

    size_t pos5 = search_str.find_last_of("aeiou");
    std::cout << "find_last_of('aeiou'): " << pos5 << " ('" << search_str[pos5] << "')\n";

    size_t pos6 = search_str.find_first_not_of("Helo ");
    std::cout << "find_first_not_of('Helo '): " << pos6 << " ('" << search_str[pos6] << "')\n";

    // 7. String Comparison
    std::cout << "\n7. STRING COMPARISON:\n";
    std::string a = "apple";
    std::string b = "banana";
    std::string c = "apple";

    std::cout << "a == b: " << (a == b) << "\n";
    std::cout << "a == c: " << (a == c) << "\n";
    std::cout << "a < b: " << (a < b) << "\n";

    std::cout << "a.compare(b): " << a.compare(b) << "\n";
    std::cout << "a.compare(c): " << a.compare(c) << "\n";

    // 8. String Concatenation
    std::cout << "\n8. STRING CONCATENATION:\n";
    std::string first = "Hello";
    std::string second = "World";

    std::string concat1 = first + " " + second;
    std::cout << "Using +: " << concat1 << "\n";

    std::string concat2 = first;
    concat2.append(" ").append(second);
    std::cout << "Using append: " << concat2 << "\n";

    // 9. String to Number Conversion
    std::cout << "\n9. STRING TO NUMBER:\n";
    std::string num_str1 = "42";
    std::string num_str2 = "3.14159";
    std::string num_str3 = "100 apples";

    int i = std::stoi(num_str1);
    double d = std::stod(num_str2);
    size_t idx;
    int i2 = std::stoi(num_str3, &idx);

    std::cout << "stoi('42'): " << i << "\n";
    std::cout << "stod('3.14159'): " << d << "\n";
    std::cout << "stoi('100 apples'): " << i2 << ", converted " << idx << " chars\n";

    // 10. Number to String Conversion
    std::cout << "\n10. NUMBER TO STRING:\n";
    int num = 42;
    double pi = 3.14159;

    std::string str_from_int = std::to_string(num);
    std::string str_from_double = std::to_string(pi);

    std::cout << "to_string(42): " << str_from_int << "\n";
    std::cout << "to_string(3.14159): " << str_from_double << "\n";

    // 11. String Algorithms
    std::cout << "\n11. STRING ALGORITHMS:\n";
    std::string algo_str = "hello world";

    // To uppercase
    std::string upper = algo_str;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    std::cout << "Uppercase: " << upper << "\n";

    // To lowercase
    std::string lower = upper;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::cout << "Lowercase: " << lower << "\n";

    // Reverse
    std::string reversed = algo_str;
    std::reverse(reversed.begin(), reversed.end());
    std::cout << "Reversed: " << reversed << "\n";

    // Remove spaces
    std::string no_spaces = "h e l l o";
    no_spaces.erase(std::remove(no_spaces.begin(), no_spaces.end(), ' '), no_spaces.end());
    std::cout << "Remove spaces: " << no_spaces << "\n";

    // 12. String Streams
    std::cout << "\n12. STRING STREAMS:\n";
    std::ostringstream oss;
    oss << "Number: " << 42 << ", Pi: " << 3.14;
    std::string from_stream = oss.str();
    std::cout << "ostringstream: " << from_stream << "\n";

    std::istringstream iss("10 20 30");
    int x, y, z;
    iss >> x >> y >> z;
    std::cout << "istringstream: x=" << x << ", y=" << y << ", z=" << z << "\n";

    // ========== STRING_VIEW ==========
    separator("STRING_VIEW (C++17)");

    // 13. string_view Basics
    std::cout << "\n13. STRING_VIEW BASICS:\n";
    std::string_view sv1 = "Hello";
    std::string str_for_view = "World";
    std::string_view sv2 = str_for_view;

    std::cout << "sv1: " << sv1 << "\n";
    std::cout << "sv2: " << sv2 << "\n";
    std::cout << "sv1.size(): " << sv1.size() << "\n";

    // 14. string_view Operations
    std::cout << "\n14. STRING_VIEW OPERATIONS:\n";
    std::string_view sv = "Hello, World!";

    std::cout << "sv[0]: " << sv[0] << "\n";
    std::cout << "sv.front(): " << sv.front() << "\n";
    std::cout << "sv.back(): " << sv.back() << "\n";

    // Substring
    std::string_view sub_sv = sv.substr(7, 5);
    std::cout << "substr(7, 5): " << sub_sv << "\n";

    // Remove prefix/suffix
    std::string_view trimmed = sv;
    trimmed.remove_prefix(7);  // Remove "Hello, "
    trimmed.remove_suffix(1);  // Remove "!"
    std::cout << "After remove_prefix/suffix: " << trimmed << "\n";

    // 15. string_view Benefits
    std::cout << "\n15. STRING_VIEW BENEFITS:\n";

    auto process_string = [](std::string_view sv) {
        std::cout << "Processing: " << sv << " (no copy!)\n";
    };

    process_string("Literal");           // No temporary string
    process_string(std::string("Temp")); // No copy

    std::string long_str = "This is a very long string";
    process_string(long_str);            // No copy

    // 16. string_view Caution
    std::cout << "\n16. STRING_VIEW CAUTION:\n";
    std::string_view dangling;
    {
        std::string temp = "Temporary";
        dangling = temp;  // Dangles after scope!
    }
    // Don't use 'dangling' here - undefined behavior!
    std::cout << "Be careful with lifetime!\n";

    // ========== PRACTICAL EXAMPLES ==========
    separator("PRACTICAL EXAMPLES");

    // 17. Trim Function
    std::cout << "\n17. TRIM FUNCTION:\n";
    auto trim = [](std::string s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
        return s;
    };

    std::string padded = "   Hello World   ";
    std::cout << "Before trim: '" << padded << "'\n";
    std::cout << "After trim: '" << trim(padded) << "'\n";

    // 18. Split String
    std::cout << "\n18. SPLIT STRING:\n";
    std::string csv = "apple,banana,cherry,date";
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;

    while ((end = csv.find(',', start)) != std::string::npos) {
        tokens.push_back(csv.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(csv.substr(start));

    std::cout << "Tokens: ";
    for (const auto& token : tokens) {
        std::cout << token << " | ";
    }
    std::cout << "\n";

    // 19. Join Strings
    std::cout << "\n19. JOIN STRINGS:\n";
    std::vector<std::string> words = {"Hello", "World", "From", "C++"};
    std::string joined;
    for (size_t i = 0; i < words.size(); ++i) {
        joined += words[i];
        if (i < words.size() - 1) joined += " ";
    }
    std::cout << "Joined: " << joined << "\n";

    // 20. Check Prefix/Suffix
    std::cout << "\n20. CHECK PREFIX/SUFFIX:\n";
    std::string filename = "document.pdf";

    bool starts_with = filename.substr(0, 3) == "doc";
    bool ends_with = filename.size() >= 4 && filename.substr(filename.size() - 4) == ".pdf";

    std::cout << "Starts with 'doc': " << (starts_with ? "yes" : "no") << "\n";
    std::cout << "Ends with '.pdf': " << (ends_with ? "yes" : "no") << "\n";

    std::cout << "\n=== END OF STRINGS ===\n";

    return 0;
}
