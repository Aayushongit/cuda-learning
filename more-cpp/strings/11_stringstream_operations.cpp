/**
 * 11_stringstream_operations.cpp
 *
 * String Stream Operations
 *
 * LEARNING OBJECTIVES:
 * - Use stringstream for string building
 * - Parse strings with stringstream
 * - Format output with stringstream
 * - Extract multiple values from strings
 * - Understand when to use stringstream
 */

#include <iostream>
#include <sstream>   // For stringstream
#include <string>
#include <iomanip>   // For formatting (setw, setprecision, etc.)
#include <vector>

int main() {
    std::cout << "=== STRINGSTREAM OPERATIONS ===" << std::endl;

    // 1. BASIC STRINGSTREAM - Building strings
    std::cout << "\n1. BUILDING STRINGS WITH STRINGSTREAM:" << std::endl;
    std::stringstream ss;
    ss << "Hello";
    ss << " ";
    ss << "World";
    ss << "!";

    std::string result = ss.str();  // Get the complete string
    std::cout << "   Result: " << result << std::endl;

    // 2. BUILDING STRINGS WITH MIXED TYPES
    std::cout << "\n2. MIXED TYPES:" << std::endl;
    std::stringstream builder;
    std::string name = "Alice";
    int age = 25;
    double height = 5.7;

    builder << "Name: " << name << ", Age: " << age << ", Height: " << height << "ft";
    std::cout << "   " << builder.str() << std::endl;

    // 3. PARSING STRINGS
    std::cout << "\n3. PARSING STRINGS:" << std::endl;
    std::string data = "42 3.14 Hello";
    std::stringstream parser(data);

    int number;
    double decimal;
    std::string word;

    parser >> number >> decimal >> word;
    std::cout << "   Original: \"" << data << "\"" << std::endl;
    std::cout << "   Number: " << number << std::endl;
    std::cout << "   Decimal: " << decimal << std::endl;
    std::cout << "   Word: " << word << std::endl;

    // 4. CLEARING AND REUSING STRINGSTREAM
    std::cout << "\n4. REUSING STRINGSTREAM:" << std::endl;
    std::stringstream reusable;

    reusable << "First use";
    std::cout << "   First: " << reusable.str() << std::endl;

    reusable.str("");  // Clear the content
    reusable.clear();  // Clear state flags

    reusable << "Second use";
    std::cout << "   Second: " << reusable.str() << std::endl;

    // 5. FORMATTING NUMBERS
    std::cout << "\n5. FORMATTING NUMBERS:" << std::endl;
    std::stringstream formatter;
    double pi = 3.14159265359;

    formatter << std::fixed << std::setprecision(2) << pi;
    std::cout << "   Pi (2 decimals): " << formatter.str() << std::endl;

    formatter.str("");
    formatter << std::fixed << std::setprecision(4) << pi;
    std::cout << "   Pi (4 decimals): " << formatter.str() << std::endl;

    // 6. WIDTH AND PADDING
    std::cout << "\n6. WIDTH AND PADDING:" << std::endl;
    std::stringstream padded;

    padded << std::setw(10) << std::setfill('*') << "Hi";
    std::cout << "   Padded: |" << padded.str() << "|" << std::endl;

    // 7. HEXADECIMAL AND OCTAL
    std::cout << "\n7. DIFFERENT NUMBER BASES:" << std::endl;
    int value = 255;
    std::stringstream hex_stream, oct_stream;

    hex_stream << std::hex << value;
    oct_stream << std::oct << value;

    std::cout << "   Decimal: " << value << std::endl;
    std::cout << "   Hex: 0x" << hex_stream.str() << std::endl;
    std::cout << "   Octal: 0" << oct_stream.str() << std::endl;

    // 8. BOOLEAN FORMATTING
    std::cout << "\n8. BOOLEAN FORMATTING:" << std::endl;
    bool flag = true;
    std::stringstream bool_stream;

    bool_stream << flag;
    std::cout << "   Default: " << bool_stream.str() << std::endl;

    bool_stream.str("");
    bool_stream << std::boolalpha << flag;
    std::cout << "   Alphabetic: " << bool_stream.str() << std::endl;

    // 9. READING LINE BY LINE
    std::cout << "\n9. READING LINE BY LINE:" << std::endl;
    std::string multiline = "Line 1\nLine 2\nLine 3";
    std::stringstream line_reader(multiline);
    std::string line;

    int line_num = 1;
    while (std::getline(line_reader, line)) {
        std::cout << "   Line " << line_num++ << ": " << line << std::endl;
    }

    // 10. PARSING CSV DATA
    std::cout << "\n10. PARSING CSV DATA:" << std::endl;
    std::string csv = "Alice,30,Engineer";
    std::stringstream csv_parser(csv);
    std::string csv_name, csv_job;
    int csv_age;

    std::getline(csv_parser, csv_name, ',');
    csv_parser >> csv_age;
    csv_parser.ignore();  // Skip comma
    std::getline(csv_parser, csv_job);

    std::cout << "   Name: " << csv_name << std::endl;
    std::cout << "   Age: " << csv_age << std::endl;
    std::cout << "   Job: " << csv_job << std::endl;

    // 11. COUNTING WORDS IN STRING
    std::cout << "\n11. COUNTING WORDS:" << std::endl;
    std::string sentence = "This is a sample sentence with words";
    std::stringstream word_counter(sentence);
    std::string temp_word;
    int word_count = 0;

    while (word_counter >> temp_word) {
        word_count++;
    }
    std::cout << "   Sentence: \"" << sentence << "\"" << std::endl;
    std::cout << "   Word count: " << word_count << std::endl;

    // 12. READING ALL WORDS INTO VECTOR
    std::cout << "\n12. EXTRACTING ALL WORDS:" << std::endl;
    std::string text = "C++ is a powerful language";
    std::stringstream word_extractor(text);
    std::vector<std::string> words;
    std::string single_word;

    while (word_extractor >> single_word) {
        words.push_back(single_word);
    }

    std::cout << "   Words extracted: ";
    for (const auto& w : words) {
        std::cout << "'" << w << "' ";
    }
    std::cout << std::endl;

    // 13. BUILDING TABLE-LIKE OUTPUT
    std::cout << "\n13. TABLE FORMATTING:" << std::endl;
    std::stringstream table;

    table << std::left;  // Left-align
    table << std::setw(15) << "Name"
          << std::setw(10) << "Age"
          << std::setw(15) << "City" << "\n";

    table << std::setw(15) << "Alice"
          << std::setw(10) << 25
          << std::setw(15) << "New York" << "\n";

    table << std::setw(15) << "Bob"
          << std::setw(10) << 30
          << std::setw(15) << "London";

    std::cout << table.str() << std::endl;

    // 14. CHECKING STREAM STATE
    std::cout << "\n14. STREAM STATE CHECKING:" << std::endl;
    std::stringstream state_check("123 abc");
    int extracted_num;

    state_check >> extracted_num;
    std::cout << "   Extracted: " << extracted_num << std::endl;
    std::cout << "   Stream good: " << (state_check.good() ? "Yes" : "No") << std::endl;

    std::string extracted_str;
    state_check >> extracted_str;
    std::cout << "   Extracted: " << extracted_str << std::endl;

    // Try to extract another number (will fail)
    int another_num;
    state_check >> another_num;
    std::cout << "   Stream good: " << (state_check.good() ? "Yes" : "No") << std::endl;
    std::cout << "   Stream fail: " << (state_check.fail() ? "Yes" : "No") << std::endl;

    // 15. PRACTICAL EXAMPLE - Log message builder
    std::cout << "\n15. PRACTICAL EXAMPLE (Log builder):" << std::endl;

    auto create_log = [](const std::string& level, const std::string& message) {
        std::stringstream log;
        log << "[" << level << "] ";
        log << "Message: " << message;
        return log.str();
    };

    std::cout << "   " << create_log("INFO", "Application started") << std::endl;
    std::cout << "   " << create_log("WARNING", "Low memory") << std::endl;
    std::cout << "   " << create_log("ERROR", "File not found") << std::endl;

    return 0;
}

/**
 * STRINGSTREAM OPERATIONS SUMMARY:
 *
 * KEY CLASSES:
 * - stringstream:  Read and write (most common)
 * - istringstream: Input only (reading from string)
 * - ostringstream: Output only (building string)
 *
 * BASIC OPERATIONS:
 *
 * 1. WRITING TO STREAM:
 *    stringstream ss;
 *    ss << value1 << " " << value2;
 *
 * 2. READING FROM STREAM:
 *    stringstream ss(input_string);
 *    ss >> var1 >> var2;
 *
 * 3. GET FINAL STRING:
 *    string result = ss.str();
 *
 * 4. SET NEW CONTENT:
 *    ss.str("new content");
 *
 * 5. CLEAR STREAM:
 *    ss.str("");      // Clear content
 *    ss.clear();      // Clear state flags
 *
 * FORMATTING MANIPULATORS:
 *
 * MANIPULATOR        | PURPOSE
 * -------------------|----------------------------------
 * setw(n)            | Set field width
 * setfill(c)         | Set fill character
 * setprecision(n)    | Set decimal precision
 * fixed              | Fixed-point notation
 * scientific         | Scientific notation
 * hex                | Hexadecimal output
 * oct                | Octal output
 * dec                | Decimal output (default)
 * boolalpha          | true/false instead of 1/0
 * left               | Left-align
 * right              | Right-align
 * internal           | Internal alignment
 *
 * COMMON USE CASES:
 *
 * 1. Building complex strings:
 *    stringstream ss;
 *    ss << "User: " << name << ", Score: " << score;
 *    string message = ss.str();
 *
 * 2. Parsing delimited data:
 *    stringstream ss(data);
 *    string token;
 *    while (getline(ss, token, ',')) {
 *        // Process token
 *    }
 *
 * 3. Type conversion:
 *    stringstream ss;
 *    ss << 123;
 *    string num_str = ss.str();  // "123"
 *
 * 4. Number formatting:
 *    stringstream ss;
 *    ss << fixed << setprecision(2) << 3.14159;
 *    // Result: "3.14"
 *
 * STREAM STATE:
 * - good():  Stream is ready
 * - eof():   End of stream reached
 * - fail():  Operation failed
 * - bad():   Fatal error
 * - clear(): Reset state flags
 *
 * ADVANTAGES OF STRINGSTREAM:
 * 1. Type-safe (unlike sprintf)
 * 2. No buffer overflow risk
 * 3. Automatic formatting
 * 4. Works with custom types (via operator<<)
 * 5. Composable and flexible
 *
 * DISADVANTAGES:
 * 1. Slower than direct string operations
 * 2. More verbose for simple cases
 * 3. State management can be tricky
 *
 * WHEN TO USE:
 * - Complex string building with mixed types
 * - Parsing structured data
 * - Number formatting
 * - When you need stream manipulators
 *
 * WHEN NOT TO USE:
 * - Simple concatenation (use + or +=)
 * - Single type conversion (use to_string/stoi)
 * - Performance-critical code
 *
 * PERFORMANCE TIP:
 * Reuse stringstream objects:
 *   stringstream ss;
 *   for (...) {
 *       ss.str("");
 *       ss.clear();
 *       ss << ...;
 *   }
 *
 * COMPILE AND RUN:
 * g++ 11_stringstream_operations.cpp -o sstream
 * ./sstream
 */
