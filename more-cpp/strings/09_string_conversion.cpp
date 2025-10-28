/**
 * 09_string_conversion.cpp
 *
 * String to Number and Number to String Conversions
 *
 * LEARNING OBJECTIVES:
 * - Convert strings to numbers (int, double, etc.)
 * - Convert numbers to strings
 * - Handle conversion errors
 * - Use C++11 conversion functions
 * - Understand older C-style conversions
 */

#include <iostream>
#include <string>
#include <sstream>   // For stringstream
#include <stdexcept> // For exception handling

int main() {
    std::cout << "=== STRING CONVERSION ===" << std::endl;

    // 1. STRING TO INT (C++11)
    std::cout << "\n1. STRING TO INT (stoi):" << std::endl;
    std::string num_str = "12345";
    int number = std::stoi(num_str);
    std::cout << "   String: \"" << num_str << "\"" << std::endl;
    std::cout << "   Integer: " << number << std::endl;
    std::cout << "   Double it: " << number * 2 << std::endl;

    // 2. STRING TO LONG (C++11)
    std::cout << "\n2. STRING TO LONG (stol, stoll):" << std::endl;
    std::string long_str = "9876543210";
    long long big_number = std::stoll(long_str);
    std::cout << "   String: \"" << long_str << "\"" << std::endl;
    std::cout << "   Long long: " << big_number << std::endl;

    // 3. STRING TO FLOAT/DOUBLE (C++11)
    std::cout << "\n3. STRING TO FLOAT/DOUBLE:" << std::endl;
    std::string float_str = "3.14159";
    double pi = std::stod(float_str);
    float pi_float = std::stof(float_str);

    std::cout << "   String: \"" << float_str << "\"" << std::endl;
    std::cout << "   Double: " << pi << std::endl;
    std::cout << "   Float: " << pi_float << std::endl;

    // 4. INT TO STRING (C++11)
    std::cout << "\n4. NUMBER TO STRING (to_string):" << std::endl;
    int age = 25;
    std::string age_str = std::to_string(age);
    std::cout << "   Integer: " << age << std::endl;
    std::cout << "   String: \"" << age_str << "\"" << std::endl;

    // 5. DOUBLE TO STRING
    std::cout << "\n5. DOUBLE TO STRING:" << std::endl;
    double price = 19.99;
    std::string price_str = std::to_string(price);
    std::cout << "   Double: " << price << std::endl;
    std::cout << "   String: \"" << price_str << "\"" << std::endl;

    // 6. HANDLING CONVERSION ERRORS
    std::cout << "\n6. HANDLING CONVERSION ERRORS:" << std::endl;
    std::string invalid = "abc123";
    std::cout << "   Trying to convert: \"" << invalid << "\"" << std::endl;

    try {
        int result = std::stoi(invalid);
        std::cout << "   Result: " << result << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "   Error: Invalid argument - " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "   Error: Out of range - " << e.what() << std::endl;
    }

    // 7. PARTIAL CONVERSION
    std::cout << "\n7. PARTIAL CONVERSION:" << std::endl;
    std::string partial = "123abc";
    size_t pos;
    int partial_num = std::stoi(partial, &pos);
    std::cout << "   String: \"" << partial << "\"" << std::endl;
    std::cout << "   Converted: " << partial_num << std::endl;
    std::cout << "   Stopped at position: " << pos << std::endl;
    std::cout << "   Remaining: \"" << partial.substr(pos) << "\"" << std::endl;

    // 8. DIFFERENT BASES (binary, octal, hex)
    std::cout << "\n8. CONVERSION WITH DIFFERENT BASES:" << std::endl;
    std::string binary = "1010";
    std::string octal = "17";
    std::string hex = "FF";

    int from_binary = std::stoi(binary, nullptr, 2);    // Base 2
    int from_octal = std::stoi(octal, nullptr, 8);      // Base 8
    int from_hex = std::stoi(hex, nullptr, 16);         // Base 16

    std::cout << "   Binary \"" << binary << "\" = " << from_binary << " (decimal)" << std::endl;
    std::cout << "   Octal \"" << octal << "\" = " << from_octal << " (decimal)" << std::endl;
    std::cout << "   Hex \"" << hex << "\" = " << from_hex << " (decimal)" << std::endl;

    // 9. USING STRINGSTREAM (alternative method)
    std::cout << "\n9. USING STRINGSTREAM:" << std::endl;
    std::string ss_num = "456";
    std::stringstream ss(ss_num);
    int ss_result;
    ss >> ss_result;
    std::cout << "   String: \"" << ss_num << "\"" << std::endl;
    std::cout << "   Integer: " << ss_result << std::endl;

    // 10. NUMBER TO STRING WITH STRINGSTREAM
    std::cout << "\n10. NUMBER TO STRING WITH STRINGSTREAM:" << std::endl;
    int value = 789;
    std::stringstream converter;
    converter << value;
    std::string value_str = converter.str();
    std::cout << "   Integer: " << value << std::endl;
    std::cout << "   String: \"" << value_str << "\"" << std::endl;

    // 11. MULTIPLE VALUES WITH STRINGSTREAM
    std::cout << "\n11. BUILDING STRING WITH MULTIPLE VALUES:" << std::endl;
    std::string product = "Laptop";
    int quantity = 3;
    double unit_price = 999.99;

    std::stringstream order;
    order << "Product: " << product << ", Qty: " << quantity
          << ", Price: $" << unit_price;

    std::cout << "   " << order.str() << std::endl;

    // 12. PARSING STRING WITH MULTIPLE NUMBERS
    std::cout << "\n12. PARSING MULTIPLE NUMBERS:" << std::endl;
    std::string coordinates = "10 20 30";
    std::stringstream coord_stream(coordinates);
    int x, y, z;
    coord_stream >> x >> y >> z;
    std::cout << "   String: \"" << coordinates << "\"" << std::endl;
    std::cout << "   x=" << x << ", y=" << y << ", z=" << z << std::endl;

    // 13. CHECKING IF STRING IS NUMERIC
    std::cout << "\n13. CHECKING IF STRING IS NUMERIC:" << std::endl;
    auto is_numeric = [](const std::string& str) {
        try {
            std::stod(str);
            return true;
        } catch (...) {
            return false;
        }
    };

    std::string test1 = "12345";
    std::string test2 = "abc";
    std::string test3 = "12.34";

    std::cout << "   \"" << test1 << "\" is numeric: " << (is_numeric(test1) ? "Yes" : "No") << std::endl;
    std::cout << "   \"" << test2 << "\" is numeric: " << (is_numeric(test2) ? "Yes" : "No") << std::endl;
    std::cout << "   \"" << test3 << "\" is numeric: " << (is_numeric(test3) ? "Yes" : "No") << std::endl;

    // 14. PRACTICAL EXAMPLE - Calculator
    std::cout << "\n14. PRACTICAL EXAMPLE (Simple calculator):" << std::endl;
    std::string input = "15 + 25";
    std::stringstream calc(input);
    int num1, num2;
    char operation;
    calc >> num1 >> operation >> num2;

    int calc_result;
    switch (operation) {
        case '+': calc_result = num1 + num2; break;
        case '-': calc_result = num1 - num2; break;
        case '*': calc_result = num1 * num2; break;
        case '/': calc_result = num1 / num2; break;
        default: calc_result = 0;
    }

    std::cout << "   Input: " << input << std::endl;
    std::cout << "   Result: " << calc_result << std::endl;

    // 15. FORMATTING NUMBERS
    std::cout << "\n15. FORMATTING NUMBERS:" << std::endl;
    double money = 1234.567;
    std::stringstream formatter;
    formatter.precision(2);
    formatter << std::fixed << money;
    std::cout << "   Double: " << money << std::endl;
    std::cout << "   Formatted: $" << formatter.str() << std::endl;

    return 0;
}

/**
 * CONVERSION FUNCTIONS SUMMARY (C++11):
 *
 * STRING TO NUMBER:
 * Function    | Type           | Example
 * ------------|----------------|---------------------------
 * stoi()      | int            | int n = stoi("123");
 * stol()      | long           | long n = stol("123456");
 * stoll()     | long long      | long long n = stoll("123");
 * stoul()     | unsigned long  | unsigned long n = stoul("123");
 * stoull()    | unsigned long long | auto n = stoull("123");
 * stof()      | float          | float f = stof("3.14");
 * stod()      | double         | double d = stod("3.14");
 * stold()     | long double    | long double d = stold("3.14");
 *
 * NUMBER TO STRING:
 * to_string(number) - Works for all numeric types
 *
 * CONVERSION PARAMETERS:
 * stoi(str, pos, base)
 * - str: String to convert
 * - pos: Pointer to store position where parsing stopped (optional)
 * - base: Number base (2-36), default is 10
 *
 * EXCEPTIONS:
 * - invalid_argument: If no conversion could be performed
 * - out_of_range: If converted value would fall out of range
 *
 * ALTERNATIVE METHODS:
 *
 * 1. STRINGSTREAM (older, more verbose):
 *    stringstream ss(str);
 *    int n;
 *    ss >> n;
 *
 * 2. C-STYLE (deprecated, avoid):
 *    atoi(str.c_str())  // String to int
 *    atof(str.c_str())  // String to float
 *    sprintf(buf, "%d", num)  // Number to string
 *
 * BEST PRACTICES:
 * 1. Use C++11 functions (stoi, to_string) for simple conversions
 * 2. Use stringstream for complex formatting/parsing
 * 3. Always handle exceptions when converting user input
 * 4. Check conversion success with try-catch or by checking position
 * 5. Use nullptr for pos parameter if you don't need it
 *
 * COMMON USE CASES:
 * - Reading numbers from user input (strings)
 * - Building strings with mixed text and numbers
 * - Parsing configuration files
 * - Converting between different number bases
 * - Data serialization/deserialization
 *
 * PERFORMANCE NOTES:
 * - to_string() is generally fast
 * - stoi() family is faster than stringstream
 * - For repeated conversions, reuse stringstream
 *
 * COMPILE AND RUN:
 * g++ 09_string_conversion.cpp -o convert
 * ./convert
 */
