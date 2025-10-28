/**
 * 13_string_splitting.cpp
 *
 * String Splitting and Tokenization
 *
 * LEARNING OBJECTIVES:
 * - Split strings by delimiters
 * - Tokenize strings
 * - Parse CSV and similar formats
 * - Handle multiple delimiters
 * - Store results in containers
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

int main() {
    std::cout << "=== STRING SPLITTING AND TOKENIZATION ===" << std::endl;

    // 1. BASIC SPLITTING WITH STRINGSTREAM
    std::cout << "\n1. BASIC SPLITTING (by whitespace):" << std::endl;
    std::string sentence = "This is a test sentence";
    std::vector<std::string> words;
    std::stringstream ss(sentence);
    std::string word;

    while (ss >> word) {
        words.push_back(word);
    }

    std::cout << "   Original: \"" << sentence << "\"" << std::endl;
    std::cout << "   Words: ";
    for (const auto& w : words) {
        std::cout << "[" << w << "] ";
    }
    std::cout << std::endl;

    // 2. SPLIT BY CUSTOM DELIMITER
    std::cout << "\n2. SPLIT BY DELIMITER (comma):" << std::endl;
    std::string csv = "apple,banana,orange,grape";
    std::vector<std::string> fruits;
    std::stringstream csv_ss(csv);
    std::string fruit;

    while (std::getline(csv_ss, fruit, ',')) {
        fruits.push_back(fruit);
    }

    std::cout << "   CSV: \"" << csv << "\"" << std::endl;
    std::cout << "   Fruits: ";
    for (const auto& f : fruits) {
        std::cout << "[" << f << "] ";
    }
    std::cout << std::endl;

    // 3. SPLIT FUNCTION IMPLEMENTATION
    std::cout << "\n3. REUSABLE SPLIT FUNCTION:" << std::endl;

    auto split = [](const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    };

    std::string path = "/home/user/documents/file.txt";
    auto path_parts = split(path, '/');

    std::cout << "   Path: \"" << path << "\"" << std::endl;
    std::cout << "   Parts: ";
    for (const auto& part : path_parts) {
        if (!part.empty()) {
            std::cout << "[" << part << "] ";
        }
    }
    std::cout << std::endl;

    // 4. SPLIT WITH MULTIPLE DELIMITERS
    std::cout << "\n4. SPLIT WITH MULTIPLE DELIMITERS:" << std::endl;

    auto split_any = [](const std::string& str, const std::string& delimiters) {
        std::vector<std::string> tokens;
        size_t start = 0;
        size_t end = str.find_first_of(delimiters);

        while (end != std::string::npos) {
            if (end != start) {  // Don't add empty tokens
                tokens.push_back(str.substr(start, end - start));
            }
            start = end + 1;
            end = str.find_first_of(delimiters, start);
        }

        if (start < str.length()) {
            tokens.push_back(str.substr(start));
        }

        return tokens;
    };

    std::string mixed = "hello,world;foo:bar|baz";
    auto mixed_tokens = split_any(mixed, ",;:|");

    std::cout << "   String: \"" << mixed << "\"" << std::endl;
    std::cout << "   Tokens: ";
    for (const auto& token : mixed_tokens) {
        std::cout << "[" << token << "] ";
    }
    std::cout << std::endl;

    // 5. SPLIT AND TRIM WHITESPACE
    std::cout << "\n5. SPLIT AND TRIM:" << std::endl;

    auto trim = [](const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return std::string("");

        size_t last = str.find_last_not_of(" \t\n\r");
        return str.substr(first, last - first + 1);
    };

    auto split_and_trim = [&trim](const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            std::string trimmed = trim(token);
            if (!trimmed.empty()) {
                tokens.push_back(trimmed);
            }
        }

        return tokens;
    };

    std::string messy = "  apple  ,  banana  ,  orange  ";
    auto clean_fruits = split_and_trim(messy, ',');

    std::cout << "   Messy: \"" << messy << "\"" << std::endl;
    std::cout << "   Clean: ";
    for (const auto& f : clean_fruits) {
        std::cout << "[" << f << "] ";
    }
    std::cout << std::endl;

    // 6. SPLIT INTO KEY-VALUE PAIRS
    std::cout << "\n6. PARSE KEY-VALUE PAIRS:" << std::endl;
    std::string config = "name=John;age=25;city=Paris";
    auto pairs = split(config, ';');

    std::cout << "   Config: \"" << config << "\"" << std::endl;
    for (const auto& pair : pairs) {
        auto kv = split(pair, '=');
        if (kv.size() == 2) {
            std::cout << "   " << kv[0] << " -> " << kv[1] << std::endl;
        }
    }

    // 7. SPLIT LINES
    std::cout << "\n7. SPLIT BY NEWLINES:" << std::endl;
    std::string multiline = "Line 1\nLine 2\nLine 3\nLine 4";
    auto lines = split(multiline, '\n');

    std::cout << "   Lines found: " << lines.size() << std::endl;
    for (size_t i = 0; i < lines.size(); i++) {
        std::cout << "   Line " << (i + 1) << ": " << lines[i] << std::endl;
    }

    // 8. TOKENIZE WITH LIMIT
    std::cout << "\n8. SPLIT WITH MAXIMUM TOKENS:" << std::endl;

    auto split_max = [](const std::string& str, char delimiter, size_t max_splits) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;
        size_t count = 0;

        while (std::getline(ss, token, delimiter)) {
            tokens.push_back(token);
            count++;
            if (count >= max_splits) {
                // Add the rest as one token
                std::string rest;
                std::getline(ss, rest, '\0');  // Read until end
                if (!rest.empty()) {
                    tokens.push_back(delimiter + rest);
                }
                break;
            }
        }

        return tokens;
    };

    std::string long_csv = "a,b,c,d,e,f,g";
    auto limited = split_max(long_csv, ',', 3);

    std::cout << "   Original: \"" << long_csv << "\"" << std::endl;
    std::cout << "   First 3 tokens: ";
    for (const auto& t : limited) {
        std::cout << "[" << t << "] ";
    }
    std::cout << std::endl;

    // 9. PARSE CSV WITH QUOTED FIELDS
    std::cout << "\n9. CSV WITH QUOTES:" << std::endl;
    // Simple version - doesn't handle all edge cases
    std::string quoted_csv = "John,\"Doe, Jr.\",25";
    std::cout << "   CSV: \"" << quoted_csv << "\"" << std::endl;
    std::cout << "   (Note: Full CSV parsing needs more complex logic)" << std::endl;

    // 10. EXTRACT WORDS (ALPHANUMERIC ONLY)
    std::cout << "\n10. EXTRACT WORDS (alphanumeric):" << std::endl;
    std::string text = "Hello, World! How are you?";
    std::vector<std::string> extracted_words;
    std::string current_word;

    for (char c : text) {
        if (std::isalnum(c)) {
            current_word += c;
        } else {
            if (!current_word.empty()) {
                extracted_words.push_back(current_word);
                current_word.clear();
            }
        }
    }
    if (!current_word.empty()) {
        extracted_words.push_back(current_word);
    }

    std::cout << "   Text: \"" << text << "\"" << std::endl;
    std::cout << "   Words: ";
    for (const auto& w : extracted_words) {
        std::cout << "[" << w << "] ";
    }
    std::cout << std::endl;

    // 11. SPLIT BY SUBSTRING (not just character)
    std::cout << "\n11. SPLIT BY SUBSTRING:" << std::endl;

    auto split_by_string = [](const std::string& str, const std::string& delimiter) {
        std::vector<std::string> tokens;
        size_t start = 0;
        size_t end = str.find(delimiter);

        while (end != std::string::npos) {
            tokens.push_back(str.substr(start, end - start));
            start = end + delimiter.length();
            end = str.find(delimiter, start);
        }

        tokens.push_back(str.substr(start));
        return tokens;
    };

    std::string url = "https://www.example.com/path/to/page";
    auto url_parts = split_by_string(url, "://");

    std::cout << "   URL: \"" << url << "\"" << std::endl;
    std::cout << "   Split by '://': ";
    for (const auto& part : url_parts) {
        std::cout << "[" << part << "] ";
    }
    std::cout << std::endl;

    // 12. JOIN STRINGS (opposite of split)
    std::cout << "\n12. JOIN STRINGS:" << std::endl;

    auto join = [](const std::vector<std::string>& strings, const std::string& separator) {
        std::string result;
        for (size_t i = 0; i < strings.size(); i++) {
            result += strings[i];
            if (i < strings.size() - 1) {
                result += separator;
            }
        }
        return result;
    };

    std::vector<std::string> to_join = {"apple", "banana", "orange"};
    std::string joined = join(to_join, ", ");

    std::cout << "   Vector: [apple, banana, orange]" << std::endl;
    std::cout << "   Joined: \"" << joined << "\"" << std::endl;

    // 13. PRACTICAL EXAMPLE - Parse command line
    std::cout << "\n13. PRACTICAL EXAMPLE (Command parsing):" << std::endl;
    std::string command = "git commit -m \"Initial commit\"";
    auto cmd_parts = split(command, ' ');

    std::cout << "   Command: \"" << command << "\"" << std::endl;
    std::cout << "   Program: " << cmd_parts[0] << std::endl;
    std::cout << "   Arguments: ";
    for (size_t i = 1; i < cmd_parts.size(); i++) {
        std::cout << cmd_parts[i] << " ";
    }
    std::cout << std::endl;

    // 14. COUNT WORDS
    std::cout << "\n14. COUNT WORDS IN STRING:" << std::endl;
    std::string paragraph = "This is a sample paragraph with multiple words";
    auto word_list = split_any(paragraph, " ");
    std::cout << "   Paragraph: \"" << paragraph << "\"" << std::endl;
    std::cout << "   Word count: " << word_list.size() << std::endl;

    return 0;
}

/**
 * STRING SPLITTING TECHNIQUES SUMMARY:
 *
 * BASIC METHODS:
 *
 * 1. STRINGSTREAM (whitespace):
 *    stringstream ss(str);
 *    string word;
 *    while (ss >> word) { ... }
 *
 * 2. GETLINE (single delimiter):
 *    stringstream ss(str);
 *    string token;
 *    while (getline(ss, token, delim)) { ... }
 *
 * 3. FIND + SUBSTR (manual):
 *    size_t pos = 0;
 *    while ((pos = str.find(delim, pos)) != npos) { ... }
 *
 * COMMON SPLIT PATTERNS:
 *
 * 1. BY SINGLE CHARACTER:
 *    getline(ss, token, ',')
 *
 * 2. BY MULTIPLE DELIMITERS:
 *    find_first_of(delimiters)
 *
 * 3. BY SUBSTRING:
 *    find(delimiter_string)
 *
 * 4. BY WHITESPACE:
 *    ss >> token
 *
 * ADVANCED TECHNIQUES:
 *
 * 1. TRIM WHILE SPLITTING:
 *    - Remove leading/trailing whitespace from each token
 *
 * 2. SKIP EMPTY TOKENS:
 *    - Check if token is empty before adding
 *
 * 3. LIMIT NUMBER OF SPLITS:
 *    - Stop after N splits, keep rest as one token
 *
 * 4. HANDLE QUOTED STRINGS:
 *    - Respect quotes in CSV/command-line parsing
 *    - More complex, needs state machine
 *
 * STORAGE OPTIONS:
 *
 * 1. VECTOR:
 *    vector<string> tokens;
 *    - Most common, flexible
 *
 * 2. ARRAY (if size known):
 *    string tokens[MAX_SIZE];
 *
 * 3. PROCESS IMMEDIATELY:
 *    while (getline(ss, token, ',')) {
 *        process(token);  // Don't store
 *    }
 *
 * PERFORMANCE CONSIDERATIONS:
 *
 * 1. STRINGSTREAM:
 *    - Easy to use
 *    - Moderate performance
 *    - Good for small to medium strings
 *
 * 2. FIND + SUBSTR:
 *    - More control
 *    - Can be faster for large strings
 *    - More code
 *
 * 3. AVOID REPEATED COPYING:
 *    - Reserve vector capacity if size known
 *    - Use string_view in C++17 if possible
 *
 * COMMON PITFALLS:
 *
 * 1. Empty tokens at boundaries:
 *    "a,,b".split(',') -> ["a", "", "b"]
 *    Check and skip if needed
 *
 * 2. Trailing delimiter:
 *    "a,b,".split(',') -> ["a", "b", ""]
 *
 * 3. Leading/trailing whitespace:
 *    " a , b ".split(',') -> [" a ", " b "]
 *    Trim tokens if needed
 *
 * 4. Multiple consecutive delimiters:
 *    "a,,b" with delimiter ','
 *    Decide if empty strings are wanted
 *
 * PRACTICAL APPLICATIONS:
 *
 * - CSV parsing
 * - URL parsing
 * - Configuration file parsing
 * - Command-line argument parsing
 * - Log file processing
 * - Data extraction
 * - Word counting
 * - Path manipulation
 *
 * C++20 NOTE:
 * C++20 adds std::ranges::split_view which provides
 * a more standard way to split strings.
 *
 * COMPILE AND RUN:
 * g++ 13_string_splitting.cpp -o split
 * ./split
 */
