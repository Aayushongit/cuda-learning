/**
 * 06_string_manipulation.cpp
 *
 * String Manipulation Operations
 *
 * LEARNING OBJECTIVES:
 * - Append strings and characters
 * - Insert content at specific positions
 * - Erase/remove parts of strings
 * - Modify string content
 * - Understand push_back and pop_back
 */

#include <iostream>
#include <string>

int main() {
    std::cout << "=== STRING MANIPULATION ===" << std::endl;

    // 1. APPEND - Adding to the end
    std::cout << "\n1. APPEND OPERATIONS:" << std::endl;
    std::string message = "Hello";
    std::cout << "   Original: " << message << std::endl;

    message.append(" World");
    std::cout << "   After append(\" World\"): " << message << std::endl;

    message.append(3, '!');  // Append 3 exclamation marks
    std::cout << "   After append(3, '!'): " << message << std::endl;

    // 2. OPERATOR += (easier way to append)
    std::cout << "\n2. OPERATOR += (APPEND):" << std::endl;
    std::string greeting = "Hi";
    std::cout << "   Original: " << greeting << std::endl;

    greeting += " there";
    std::cout << "   After += \" there\": " << greeting << std::endl;

    greeting += '!';
    std::cout << "   After += '!': " << greeting << std::endl;

    // 3. PUSH_BACK - Add single character to end
    std::cout << "\n3. PUSH_BACK (single character):" << std::endl;
    std::string word = "Hel";
    std::cout << "   Original: " << word << std::endl;

    word.push_back('l');
    word.push_back('o');
    std::cout << "   After push_back('l') and push_back('o'): " << word << std::endl;

    // 4. POP_BACK - Remove last character
    std::cout << "\n4. POP_BACK (remove last character):" << std::endl;
    std::string text = "Hello!";
    std::cout << "   Original: " << text << std::endl;

    text.pop_back();
    std::cout << "   After pop_back(): " << text << std::endl;

    // 5. INSERT - Add at specific position
    std::cout << "\n5. INSERT OPERATIONS:" << std::endl;
    std::string sentence = "I C++";
    std::cout << "   Original: " << sentence << std::endl;

    sentence.insert(2, "love ");  // Insert at position 2
    std::cout << "   After insert(2, \"love \"): " << sentence << std::endl;

    // Insert single character
    sentence.insert(0, 1, '*');
    std::cout << "   After insert(0, 1, '*'): " << sentence << std::endl;

    // 6. ERASE - Remove characters
    std::cout << "\n6. ERASE OPERATIONS:" << std::endl;
    std::string data = "Hello World";
    std::cout << "   Original: " << data << std::endl;

    data.erase(5, 6);  // Erase from position 5, count 6 characters
    std::cout << "   After erase(5, 6): " << data << std::endl;

    // Erase from position to end
    data = "Hello World";
    data.erase(5);  // Erase from position 5 to end
    std::cout << "   After erase(5): " << data << std::endl;

    // 7. CLEAR - Remove all content
    std::cout << "\n7. CLEAR (remove all):" << std::endl;
    std::string temp = "This will be cleared";
    std::cout << "   Original: " << temp << std::endl;
    std::cout << "   Length: " << temp.length() << std::endl;

    temp.clear();
    std::cout << "   After clear(): '" << temp << "'" << std::endl;
    std::cout << "   Length: " << temp.length() << std::endl;

    // 8. ASSIGN - Replace entire content
    std::cout << "\n8. ASSIGN OPERATIONS:" << std::endl;
    std::string var = "Old value";
    std::cout << "   Original: " << var << std::endl;

    var.assign("New value");
    std::cout << "   After assign(\"New value\"): " << var << std::endl;

    var.assign(5, '*');  // Assign 5 asterisks
    std::cout << "   After assign(5, '*'): " << var << std::endl;

    // 9. SWAP - Exchange contents of two strings
    std::cout << "\n9. SWAP STRINGS:" << std::endl;
    std::string first = "First";
    std::string second = "Second";
    std::cout << "   Before swap:" << std::endl;
    std::cout << "   first = " << first << std::endl;
    std::cout << "   second = " << second << std::endl;

    first.swap(second);
    std::cout << "   After swap:" << std::endl;
    std::cout << "   first = " << first << std::endl;
    std::cout << "   second = " << second << std::endl;

    // 10. RESIZE - Change string size
    std::cout << "\n10. RESIZE STRING:" << std::endl;
    std::string resizable = "Hi";
    std::cout << "   Original: '" << resizable << "' (length: " << resizable.length() << ")" << std::endl;

    resizable.resize(5);  // Extend with null characters
    std::cout << "   After resize(5): '" << resizable << "' (length: " << resizable.length() << ")" << std::endl;

    resizable.resize(8, '*');  // Extend with asterisks
    std::cout << "   After resize(8, '*'): '" << resizable << "' (length: " << resizable.length() << ")" << std::endl;

    resizable.resize(3);  // Shrink
    std::cout << "   After resize(3): '" << resizable << "' (length: " << resizable.length() << ")" << std::endl;

    // 11. BUILDING STRINGS EFFICIENTLY
    std::cout << "\n11. BUILDING STRINGS:" << std::endl;
    std::string result = "";
    result.reserve(50);  // Reserve space to avoid reallocations

    result += "Building ";
    result.append("a ");
    result.push_back('s');
    result.push_back('t');
    result += "ring efficiently!";

    std::cout << "   Result: " << result << std::endl;

    // 12. REMOVING SPACES FROM STRING
    std::cout << "\n12. REMOVING SPECIFIC CHARACTERS:" << std::endl;
    std::string with_spaces = "H e l l o   W o r l d";
    std::cout << "   Original: " << with_spaces << std::endl;

    // Remove all spaces
    size_t pos;
    while ((pos = with_spaces.find(' ')) != std::string::npos) {
        with_spaces.erase(pos, 1);
    }
    std::cout << "   After removing spaces: " << with_spaces << std::endl;

    // 13. PRACTICAL EXAMPLE - Building a sentence
    std::cout << "\n13. PRACTICAL EXAMPLE:" << std::endl;
    std::string name = "Alice";
    std::string age_str = "25";
    std::string city = "Paris";

    std::string bio;
    bio.append(name);
    bio.append(" is ");
    bio.append(age_str);
    bio.append(" years old and lives in ");
    bio.append(city);
    bio.push_back('.');

    std::cout << "   Bio: " << bio << std::endl;

    return 0;
}

/**
 * STRING MANIPULATION METHODS SUMMARY:
 *
 * METHOD           | PURPOSE                      | SYNTAX EXAMPLE
 * -----------------|------------------------------|--------------------------------
 * append()         | Add to end                   | str.append(" text")
 * +=               | Add to end (easier)          | str += " text"
 * push_back()      | Add single char to end       | str.push_back('c')
 * pop_back()       | Remove last character        | str.pop_back()
 * insert()         | Insert at position           | str.insert(pos, "text")
 * erase()          | Remove characters            | str.erase(pos, count)
 * clear()          | Remove all content           | str.clear()
 * assign()         | Replace all content          | str.assign("new")
 * swap()           | Exchange with another string | str1.swap(str2)
 * resize()         | Change size                  | str.resize(new_size)
 *
 * APPEND vs CONCATENATION:
 * append():        str.append(" World")     (modifies str)
 * operator+=:      str += " World"          (modifies str, cleaner)
 * operator+:       str2 = str1 + " World"   (creates new string)
 *
 * INSERT POSITION:
 * - Position 0 = beginning
 * - Position length() = end (same as append)
 * - insert(pos, count, char) for repeated characters
 *
 * ERASE VARIANTS:
 * - erase(pos, count): Remove 'count' chars starting at 'pos'
 * - erase(pos): Remove from 'pos' to end
 * - erase(): Remove all (same as clear)
 *
 * PERFORMANCE TIPS:
 * 1. Use reserve() before multiple append operations
 * 2. Use += instead of + for appending to same string
 * 3. push_back() is fastest for single characters
 * 4. Avoid repeated insert/erase in loops (slow)
 *
 * COMMON PATTERNS:
 * 1. Building strings:
 *    string s;
 *    s.reserve(expected_size);
 *    s += part1;
 *    s += part2;
 *
 * 2. Removing characters:
 *    while ((pos = str.find(char)) != npos) {
 *        str.erase(pos, 1);
 *    }
 *
 * 3. Clearing for reuse:
 *    str.clear();
 *    // or
 *    str = "";
 *
 * COMPILE AND RUN:
 * g++ 06_string_manipulation.cpp -o manipulate
 * ./manipulate
 */
