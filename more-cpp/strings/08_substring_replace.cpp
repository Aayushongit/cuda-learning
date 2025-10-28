/**
 * 08_substring_replace.cpp
 *
 * Substring Extraction and Replace Operations
 *
 * LEARNING OBJECTIVES:
 * - Extract substrings from strings
 * - Replace parts of strings
 * - Understand substr() method
 * - Use replace() effectively
 * - Practical string transformation techniques
 */

#include <iostream>
#include <string>

int main() {
    std::cout << "=== SUBSTRING AND REPLACE OPERATIONS ===" << std::endl;

    // 1. BASIC SUBSTR - Extract substring
    std::cout << "\n1. BASIC SUBSTR:" << std::endl;
    std::string text = "Hello World";
    std::cout << "   Original: " << text << std::endl;

    // substr(start_pos, length)
    std::string sub1 = text.substr(0, 5);  // From position 0, length 5
    std::cout << "   substr(0, 5): " << sub1 << std::endl;

    std::string sub2 = text.substr(6, 5);  // From position 6, length 5
    std::cout << "   substr(6, 5): " << sub2 << std::endl;

    // 2. SUBSTR FROM POSITION TO END
    std::cout << "\n2. SUBSTR FROM POSITION TO END:" << std::endl;
    std::string sentence = "C++ Programming";
    std::cout << "   Original: " << sentence << std::endl;

    std::string from_pos = sentence.substr(4);  // From position 4 to end
    std::cout << "   substr(4): " << from_pos << std::endl;

    // 3. EXTRACTING WORDS
    std::cout << "\n3. EXTRACTING WORDS:" << std::endl;
    std::string full_name = "John Michael Doe";
    std::cout << "   Full name: " << full_name << std::endl;

    size_t first_space = full_name.find(' ');
    size_t last_space = full_name.rfind(' ');

    std::string first_name = full_name.substr(0, first_space);
    std::string last_name = full_name.substr(last_space + 1);
    std::string middle_name = full_name.substr(first_space + 1, last_space - first_space - 1);

    std::cout << "   First name: " << first_name << std::endl;
    std::cout << "   Middle name: " << middle_name << std::endl;
    std::cout << "   Last name: " << last_name << std::endl;

    // 4. BASIC REPLACE
    std::cout << "\n4. BASIC REPLACE:" << std::endl;
    std::string message = "I like Java programming";
    std::cout << "   Original: " << message << std::endl;

    // replace(start_pos, length, new_string)
    message.replace(7, 4, "C++");  // Replace 4 chars starting at position 7
    std::cout << "   After replace(7, 4, \"C++\"): " << message << std::endl;

    // 5. REPLACE ALL OCCURRENCES
    std::cout << "\n5. REPLACE ALL OCCURRENCES:" << std::endl;
    std::string document = "The cat sat on the mat. The cat was happy.";
    std::cout << "   Original: " << document << std::endl;

    std::string old_word = "cat";
    std::string new_word = "dog";

    size_t pos = 0;
    while ((pos = document.find(old_word, pos)) != std::string::npos) {
        document.replace(pos, old_word.length(), new_word);
        pos += new_word.length();  // Move past the replacement
    }
    std::cout << "   After replacing 'cat' with 'dog': " << document << std::endl;

    // 6. REPLACE WITH DIFFERENT LENGTH STRING
    std::cout << "\n6. REPLACE WITH DIFFERENT LENGTH:" << std::endl;
    std::string greeting = "Hi there!";
    std::cout << "   Original: " << greeting << std::endl;

    greeting.replace(0, 2, "Hello");  // Replace "Hi" with "Hello"
    std::cout << "   After replace: " << greeting << std::endl;

    // 7. REPLACE WITH REPEATED CHARACTERS
    std::cout << "\n7. REPLACE WITH REPEATED CHARACTERS:" << std::endl;
    std::string password = "myPassword123";
    std::cout << "   Original: " << password << std::endl;

    // Replace with asterisks
    password.replace(2, 8, 8, '*');  // Replace 8 chars with 8 asterisks
    std::cout << "   Masked: " << password << std::endl;

    // 8. EXTRACT FILE EXTENSION
    std::cout << "\n8. EXTRACT FILE EXTENSION:" << std::endl;
    std::string filename = "document.pdf";
    std::cout << "   Filename: " << filename << std::endl;

    size_t dot_pos = filename.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string extension = filename.substr(dot_pos);
        std::cout << "   Extension: " << extension << std::endl;
        std::string name = filename.substr(0, dot_pos);
        std::cout << "   Name: " << name << std::endl;
    }

    // 9. EXTRACT PATH COMPONENTS
    std::cout << "\n9. EXTRACT PATH COMPONENTS:" << std::endl;
    std::string path = "/home/user/documents/file.txt";
    std::cout << "   Path: " << path << std::endl;

    size_t last_slash = path.rfind('/');
    std::string directory = path.substr(0, last_slash);
    std::string file = path.substr(last_slash + 1);

    std::cout << "   Directory: " << directory << std::endl;
    std::cout << "   File: " << file << std::endl;

    // 10. CENSORING WORDS
    std::cout << "\n10. CENSORING WORDS:" << std::endl;
    std::string comment = "This is a bad word in the text";
    std::cout << "   Original: " << comment << std::endl;

    std::string bad_word = "bad";
    pos = comment.find(bad_word);
    if (pos != std::string::npos) {
        comment.replace(pos, bad_word.length(), std::string(bad_word.length(), '*'));
    }
    std::cout << "   Censored: " << comment << std::endl;

    // 11. EXTRACT SUBSTRING BETWEEN DELIMITERS
    std::cout << "\n11. EXTRACT BETWEEN DELIMITERS:" << std::endl;
    std::string html = "<div>Content here</div>";
    std::cout << "   HTML: " << html << std::endl;

    size_t start = html.find('>') + 1;
    size_t end = html.find('<', start);
    std::string content = html.substr(start, end - start);

    std::cout << "   Extracted content: " << content << std::endl;

    // 12. SWAP PARTS OF STRING
    std::cout << "\n12. SWAP PARTS OF STRING:" << std::endl;
    std::string name_format = "Doe, John";
    std::cout << "   Original: " << name_format << std::endl;

    size_t comma = name_format.find(',');
    std::string last = name_format.substr(0, comma);
    std::string first = name_format.substr(comma + 2);  // +2 to skip comma and space

    std::string swapped = first + " " + last;
    std::cout << "   Swapped: " << swapped << std::endl;

    // 13. PRACTICAL EXAMPLE - URL parsing
    std::cout << "\n13. PRACTICAL EXAMPLE (URL parsing):" << std::endl;
    std::string url = "https://www.example.com:8080/path/to/page";
    std::cout << "   URL: " << url << std::endl;

    // Extract protocol
    size_t protocol_end = url.find("://");
    std::string protocol = url.substr(0, protocol_end);
    std::cout << "   Protocol: " << protocol << std::endl;

    // Extract domain (with port)
    size_t domain_start = protocol_end + 3;
    size_t path_start = url.find('/', domain_start);
    std::string domain = url.substr(domain_start, path_start - domain_start);
    std::cout << "   Domain: " << domain << std::endl;

    // Extract path
    std::string url_path = url.substr(path_start);
    std::cout << "   Path: " << url_path << std::endl;

    // 14. REPLACE WITH SUBSTRING FROM ANOTHER STRING
    std::cout << "\n14. REPLACE WITH SUBSTRING:" << std::endl;
    std::string base = "Hello World";
    std::string source = "Beautiful Day";
    std::cout << "   Base: " << base << std::endl;

    // Replace "World" with "Day"
    size_t replace_pos = base.find("World");
    base.replace(replace_pos, 5, source, 10, 3);  // Take "Day" from source
    std::cout << "   After replace: " << base << std::endl;

    return 0;
}

/**
 * SUBSTR AND REPLACE SUMMARY:
 *
 * SUBSTR SYNTAX:
 * string.substr(pos, len)
 * - pos: Starting position
 * - len: Number of characters to extract (optional, default = to end)
 *
 * SUBSTR VARIATIONS:
 * substr(pos):       Extract from pos to end
 * substr(pos, len):  Extract len characters starting at pos
 *
 * REPLACE SYNTAX:
 * string.replace(pos, len, str)
 * - pos: Position to start replacing
 * - len: Number of characters to replace
 * - str: New string to insert
 *
 * REPLACE VARIATIONS:
 * replace(pos, len, str):           Replace with string
 * replace(pos, len, str, pos2, n):  Replace with substring of str
 * replace(pos, len, n, char):       Replace with n copies of char
 * replace(begin, end, str):         Replace iterator range with str
 *
 * COMMON PATTERNS:
 *
 * 1. Extract everything after delimiter:
 *    pos = str.find(delim);
 *    result = str.substr(pos + 1);
 *
 * 2. Extract between two delimiters:
 *    start = str.find(open) + 1;
 *    end = str.find(close, start);
 *    result = str.substr(start, end - start);
 *
 * 3. Replace all occurrences:
 *    while ((pos = str.find(old)) != npos) {
 *        str.replace(pos, old.length(), new_str);
 *        pos += new_str.length();
 *    }
 *
 * 4. Extract file extension:
 *    pos = filename.rfind('.');
 *    ext = filename.substr(pos);
 *
 * IMPORTANT NOTES:
 * - substr() creates a NEW string (doesn't modify original)
 * - replace() MODIFIES the original string
 * - If pos is out of range, throws out_of_range exception
 * - If len is too large, extracts until end of string
 *
 * PERFORMANCE:
 * - substr() creates a copy (O(n) where n = substring length)
 * - replace() may require reallocation if new size differs
 * - Multiple replaces can be expensive; consider building new string
 *
 * COMPILE AND RUN:
 * g++ 08_substring_replace.cpp -o substr
 * ./substr
 */
