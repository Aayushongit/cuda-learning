/**
 * 12_string_algorithms.cpp
 *
 * String Algorithms and Transformations
 *
 * LEARNING OBJECTIVES:
 * - Transform strings (uppercase, lowercase)
 * - Sort strings and characters
 * - Reverse strings
 * - Remove characters
 * - Use STL algorithms with strings
 * - Custom string operations
 */

#include <iostream>
#include <string>
#include <algorithm>  // For transform, sort, reverse, remove, etc.
#include <cctype>     // For toupper, tolower, isdigit, etc.

int main() {
    std::cout << "=== STRING ALGORITHMS ===" << std::endl;

    // 1. CONVERT TO UPPERCASE
    std::cout << "\n1. CONVERT TO UPPERCASE:" << std::endl;
    std::string text = "Hello World";
    std::cout << "   Original: " << text << std::endl;

    std::transform(text.begin(), text.end(), text.begin(), ::toupper);
    std::cout << "   Uppercase: " << text << std::endl;

    // 2. CONVERT TO LOWERCASE
    std::cout << "\n2. CONVERT TO LOWERCASE:" << std::endl;
    std::string upper_text = "HELLO WORLD";
    std::cout << "   Original: " << upper_text << std::endl;

    std::transform(upper_text.begin(), upper_text.end(), upper_text.begin(), ::tolower);
    std::cout << "   Lowercase: " << upper_text << std::endl;

    // 3. REVERSE STRING
    std::cout << "\n3. REVERSE STRING:" << std::endl;
    std::string word = "Palindrome";
    std::cout << "   Original: " << word << std::endl;

    std::reverse(word.begin(), word.end());
    std::cout << "   Reversed: " << word << std::endl;

    // 4. SORT CHARACTERS IN STRING
    std::cout << "\n4. SORT CHARACTERS:" << std::endl;
    std::string jumbled = "dcba";
    std::cout << "   Original: " << jumbled << std::endl;

    std::sort(jumbled.begin(), jumbled.end());
    std::cout << "   Sorted: " << jumbled << std::endl;

    // 5. CHECK IF STRING IS PALINDROME
    std::cout << "\n5. CHECK PALINDROME:" << std::endl;
    auto is_palindrome = [](const std::string& str) {
        std::string temp = str;
        std::transform(temp.begin(), temp.end(), temp.begin(), ::tolower);
        std::string reversed = temp;
        std::reverse(reversed.begin(), reversed.end());
        return temp == reversed;
    };

    std::string test1 = "racecar";
    std::string test2 = "hello";

    std::cout << "   \"" << test1 << "\" is palindrome: " << (is_palindrome(test1) ? "Yes" : "No") << std::endl;
    std::cout << "   \"" << test2 << "\" is palindrome: " << (is_palindrome(test2) ? "Yes" : "No") << std::endl;

    // 6. REMOVE SPECIFIC CHARACTER
    std::cout << "\n6. REMOVE SPECIFIC CHARACTER:" << std::endl;
    std::string with_spaces = "H e l l o";
    std::cout << "   Original: " << with_spaces << std::endl;

    with_spaces.erase(std::remove(with_spaces.begin(), with_spaces.end(), ' '), with_spaces.end());
    std::cout << "   Without spaces: " << with_spaces << std::endl;

    // 7. REMOVE ALL DIGITS
    std::cout << "\n7. REMOVE ALL DIGITS:" << std::endl;
    std::string mixed = "Hello123World456";
    std::cout << "   Original: " << mixed << std::endl;

    mixed.erase(std::remove_if(mixed.begin(), mixed.end(), ::isdigit), mixed.end());
    std::cout << "   Without digits: " << mixed << std::endl;

    // 8. KEEP ONLY LETTERS
    std::cout << "\n8. KEEP ONLY LETTERS:" << std::endl;
    std::string messy = "H3ll0! W0rld#123";
    std::cout << "   Original: " << messy << std::endl;

    messy.erase(std::remove_if(messy.begin(), messy.end(),
                [](char c) { return !std::isalpha(c); }), messy.end());
    std::cout << "   Letters only: " << messy << std::endl;

    // 9. COUNT SPECIFIC CHARACTER
    std::cout << "\n9. COUNT SPECIFIC CHARACTER:" << std::endl;
    std::string sentence = "hello world";
    char target = 'l';

    int count = std::count(sentence.begin(), sentence.end(), target);
    std::cout << "   String: \"" << sentence << "\"" << std::endl;
    std::cout << "   Count of '" << target << "': " << count << std::endl;

    // 10. COUNT VOWELS
    std::cout << "\n10. COUNT VOWELS:" << std::endl;
    std::string phrase = "Programming is fun";
    int vowel_count = std::count_if(phrase.begin(), phrase.end(),
                                    [](char c) {
                                        c = std::tolower(c);
                                        return c == 'a' || c == 'e' || c == 'i' ||
                                               c == 'o' || c == 'u';
                                    });
    std::cout << "   Phrase: \"" << phrase << "\"" << std::endl;
    std::cout << "   Vowel count: " << vowel_count << std::endl;

    // 11. REPLACE ALL OCCURRENCES
    std::cout << "\n11. REPLACE ALL OCCURRENCES:" << std::endl;
    std::string doc = "cat cat cat";
    std::cout << "   Original: " << doc << std::endl;

    std::replace(doc.begin(), doc.end(), 'c', 'b');
    std::cout << "   After replace 'c' with 'b': " << doc << std::endl;

    // 12. ROTATE STRING
    std::cout << "\n12. ROTATE STRING:" << std::endl;
    std::string rotatable = "ABCDEF";
    std::cout << "   Original: " << rotatable << std::endl;

    std::rotate(rotatable.begin(), rotatable.begin() + 2, rotatable.end());
    std::cout << "   Rotated left by 2: " << rotatable << std::endl;

    // 13. FIND FIRST DIGIT
    std::cout << "\n13. FIND FIRST DIGIT:" << std::endl;
    std::string alpha_num = "abc123def";
    auto it = std::find_if(alpha_num.begin(), alpha_num.end(), ::isdigit);

    if (it != alpha_num.end()) {
        std::cout << "   String: \"" << alpha_num << "\"" << std::endl;
        std::cout << "   First digit: " << *it << std::endl;
        std::cout << "   Position: " << std::distance(alpha_num.begin(), it) << std::endl;
    }

    // 14. CHECK IF ALL CHARACTERS ARE UPPERCASE
    std::cout << "\n14. CHECK IF ALL UPPERCASE:" << std::endl;
    std::string upper1 = "HELLO";
    std::string upper2 = "HeLLo";

    bool all_upper1 = std::all_of(upper1.begin(), upper1.end(), ::isupper);
    bool all_upper2 = std::all_of(upper2.begin(), upper2.end(), ::isupper);

    std::cout << "   \"" << upper1 << "\" all uppercase: " << (all_upper1 ? "Yes" : "No") << std::endl;
    std::cout << "   \"" << upper2 << "\" all uppercase: " << (all_upper2 ? "Yes" : "No") << std::endl;

    // 15. TRIM WHITESPACE (LEFT AND RIGHT)
    std::cout << "\n15. TRIM WHITESPACE:" << std::endl;
    std::string padded = "   Hello World   ";
    std::cout << "   Original: |" << padded << "|" << std::endl;

    // Trim left
    padded.erase(padded.begin(), std::find_if(padded.begin(), padded.end(),
                 [](unsigned char c) { return !std::isspace(c); }));

    // Trim right
    padded.erase(std::find_if(padded.rbegin(), padded.rend(),
                 [](unsigned char c) { return !std::isspace(c); }).base(), padded.end());

    std::cout << "   Trimmed: |" << padded << "|" << std::endl;

    // 16. FIND MAX CHARACTER
    std::cout << "\n16. FIND MAX CHARACTER:" << std::endl;
    std::string chars = "programming";
    auto max_char = std::max_element(chars.begin(), chars.end());
    std::cout << "   String: \"" << chars << "\"" << std::endl;
    std::cout << "   Max character: " << *max_char << std::endl;

    // 17. SHUFFLE STRING
    std::cout << "\n17. SHUFFLE STRING:" << std::endl;
    std::string shuffleable = "ABCDEFGH";
    std::cout << "   Original: " << shuffleable << std::endl;

    std::random_shuffle(shuffleable.begin(), shuffleable.end());
    std::cout << "   Shuffled: " << shuffleable << std::endl;

    // 18. REMOVE DUPLICATES
    std::cout << "\n18. REMOVE CONSECUTIVE DUPLICATES:" << std::endl;
    std::string duplicates = "aabbccddee";
    std::cout << "   Original: " << duplicates << std::endl;

    duplicates.erase(std::unique(duplicates.begin(), duplicates.end()), duplicates.end());
    std::cout << "   After unique: " << duplicates << std::endl;

    // 19. CAPITALIZE FIRST LETTER
    std::cout << "\n19. CAPITALIZE FIRST LETTER:" << std::endl;
    std::string lowercase = "hello world";
    std::cout << "   Original: " << lowercase << std::endl;

    if (!lowercase.empty()) {
        lowercase[0] = std::toupper(lowercase[0]);
    }
    std::cout << "   Capitalized: " << lowercase << std::endl;

    // 20. TITLE CASE (capitalize each word)
    std::cout << "\n20. TITLE CASE:" << std::endl;
    std::string title = "hello world from cpp";
    std::cout << "   Original: " << title << std::endl;

    bool new_word = true;
    for (char& c : title) {
        if (std::isspace(c)) {
            new_word = true;
        } else if (new_word) {
            c = std::toupper(c);
            new_word = false;
        }
    }
    std::cout << "   Title case: " << title << std::endl;

    return 0;
}

/**
 * STRING ALGORITHMS SUMMARY:
 *
 * KEY ALGORITHMS FROM <algorithm>:
 *
 * ALGORITHM         | PURPOSE                        | EXAMPLE
 * ------------------|--------------------------------|---------------------------
 * transform()       | Apply function to each element | toupper, tolower
 * reverse()         | Reverse order                  | Reverse string
 * sort()            | Sort elements                  | Sort characters
 * remove()          | Remove specific value          | Remove spaces
 * remove_if()       | Remove by condition            | Remove digits
 * count()           | Count occurrences              | Count character
 * count_if()        | Count by condition             | Count vowels
 * replace()         | Replace value                  | Replace character
 * rotate()          | Rotate elements                | Shift string
 * find_if()         | Find by condition              | Find first digit
 * all_of()          | Check if all match             | All uppercase?
 * any_of()          | Check if any match             | Has digit?
 * none_of()         | Check if none match            | No spaces?
 * unique()          | Remove consecutive dups        | Remove duplicates
 * max_element()     | Find maximum                   | Max character
 * min_element()     | Find minimum                   | Min character
 *
 * CHARACTER CLASSIFICATION (<cctype>):
 *
 * FUNCTION     | CHECKS IF CHARACTER IS
 * -------------|------------------------
 * isalpha()    | Letter (a-z, A-Z)
 * isdigit()    | Digit (0-9)
 * isalnum()    | Letter or digit
 * isspace()    | Whitespace
 * isupper()    | Uppercase letter
 * islower()    | Lowercase letter
 * ispunct()    | Punctuation
 * toupper()    | Convert to uppercase
 * tolower()    | Convert to lowercase
 *
 * COMMON PATTERNS:
 *
 * 1. CASE CONVERSION:
 *    transform(str.begin(), str.end(), str.begin(), ::toupper);
 *
 * 2. REMOVE-ERASE IDIOM:
 *    str.erase(remove(str.begin(), str.end(), ' '), str.end());
 *
 * 3. REMOVE WITH CONDITION:
 *    str.erase(remove_if(str.begin(), str.end(), ::isdigit), str.end());
 *
 * 4. TRIM WHITESPACE:
 *    // Left trim
 *    str.erase(str.begin(), find_if(str.begin(), str.end(),
 *              [](char c) { return !isspace(c); }));
 *    // Right trim
 *    str.erase(find_if(str.rbegin(), str.rend(),
 *              [](char c) { return !isspace(c); }).base(), str.end());
 *
 * 5. COUNT CONDITION:
 *    int n = count_if(str.begin(), str.end(), ::isdigit);
 *
 * LAMBDA FUNCTIONS WITH STRINGS:
 * Very useful for custom conditions:
 *    remove_if(str.begin(), str.end(),
 *              [](char c) { return c == 'a' || c == 'e'; })
 *
 * IMPORTANT NOTES:
 * 1. Most algorithms modify the string in-place
 * 2. remove() and remove_if() don't actually remove; use erase()
 * 3. unique() only removes consecutive duplicates; sort first if needed
 * 4. Be careful with ::toupper/::tolower vs std::toupper/std::tolower
 * 5. Use unsigned char for character classification functions
 *
 * PERFORMANCE:
 * - Most algorithms are O(n) where n is string length
 * - sort() is O(n log n)
 * - Chaining operations can be expensive; combine when possible
 *
 * BEST PRACTICES:
 * 1. Use algorithms instead of manual loops when possible
 * 2. Combine operations to avoid multiple passes
 * 3. Use lambda functions for custom conditions
 * 4. Remember the remove-erase idiom
 * 5. Check for empty strings before operations
 *
 * COMPILE AND RUN:
 * g++ 12_string_algorithms.cpp -o algorithms
 * ./algorithms
 */
