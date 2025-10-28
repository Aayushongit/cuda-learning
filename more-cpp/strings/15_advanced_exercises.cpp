/**
 * 15_advanced_exercises.cpp
 *
 * Advanced String Operations and Practice Exercises
 *
 * LEARNING OBJECTIVES:
 * - Apply all string concepts learned
 * - Solve real-world string problems
 * - Practice advanced string manipulation
 * - Combine multiple techniques
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <map>

int main() {
    std::cout << "=== ADVANCED STRING EXERCISES ===" << std::endl;

    // EXERCISE 1: Anagram Checker
    std::cout << "\n1. ANAGRAM CHECKER:" << std::endl;
    auto is_anagram = [](std::string s1, std::string s2) {
        if (s1.length() != s2.length()) return false;

        std::sort(s1.begin(), s1.end());
        std::sort(s2.begin(), s2.end());

        return s1 == s2;
    };

    std::cout << "   'listen' and 'silent': " << (is_anagram("listen", "silent") ? "Anagram" : "Not anagram") << std::endl;
    std::cout << "   'hello' and 'world': " << (is_anagram("hello", "world") ? "Anagram" : "Not anagram") << std::endl;

    // EXERCISE 2: Word Frequency Counter
    std::cout << "\n2. WORD FREQUENCY COUNTER:" << std::endl;
    std::string text = "the quick brown fox jumps over the lazy dog the fox";

    std::map<std::string, int> word_count;
    std::stringstream ss(text);
    std::string word;

    while (ss >> word) {
        word_count[word]++;
    }

    std::cout << "   Text: \"" << text << "\"" << std::endl;
    std::cout << "   Word frequencies:" << std::endl;
    for (const auto& pair : word_count) {
        std::cout << "   '" << pair.first << "': " << pair.second << std::endl;
    }

    // EXERCISE 3: Longest Common Prefix
    std::cout << "\n3. LONGEST COMMON PREFIX:" << std::endl;
    auto longest_common_prefix = [](const std::vector<std::string>& strings) {
        if (strings.empty()) return std::string("");

        std::string prefix = strings[0];

        for (size_t i = 1; i < strings.size(); i++) {
            while (strings[i].find(prefix) != 0) {
                prefix = prefix.substr(0, prefix.length() - 1);
                if (prefix.empty()) return std::string("");
            }
        }

        return prefix;
    };

    std::vector<std::string> words = {"flower", "flow", "flight"};
    std::cout << "   Words: flower, flow, flight" << std::endl;
    std::cout << "   Common prefix: \"" << longest_common_prefix(words) << "\"" << std::endl;

    // EXERCISE 4: Valid Parentheses Checker
    std::cout << "\n4. VALID PARENTHESES CHECKER:" << std::endl;
    auto is_valid_parentheses = [](const std::string& s) {
        std::vector<char> stack;

        for (char c : s) {
            if (c == '(' || c == '[' || c == '{') {
                stack.push_back(c);
            } else {
                if (stack.empty()) return false;

                char top = stack.back();
                if ((c == ')' && top == '(') ||
                    (c == ']' && top == '[') ||
                    (c == '}' && top == '{')) {
                    stack.pop_back();
                } else {
                    return false;
                }
            }
        }

        return stack.empty();
    };

    std::string test1 = "()[]{}";
    std::string test2 = "([)]";
    std::cout << "   \"" << test1 << "\": " << (is_valid_parentheses(test1) ? "Valid" : "Invalid") << std::endl;
    std::cout << "   \"" << test2 << "\": " << (is_valid_parentheses(test2) ? "Valid" : "Invalid") << std::endl;

    // EXERCISE 5: String Compression
    std::cout << "\n5. STRING COMPRESSION:" << std::endl;
    auto compress_string = [](const std::string& s) {
        if (s.empty()) return std::string("");

        std::string result;
        int count = 1;

        for (size_t i = 1; i <= s.length(); i++) {
            if (i < s.length() && s[i] == s[i - 1]) {
                count++;
            } else {
                result += s[i - 1];
                if (count > 1) {
                    result += std::to_string(count);
                }
                count = 1;
            }
        }

        return result.length() < s.length() ? result : s;
    };

    std::string original = "aabcccccaaa";
    std::string compressed = compress_string(original);
    std::cout << "   Original: " << original << std::endl;
    std::cout << "   Compressed: " << compressed << std::endl;

    // EXERCISE 6: Remove Duplicates
    std::cout << "\n6. REMOVE ALL DUPLICATES:" << std::endl;
    auto remove_duplicates = [](std::string s) {
        std::string result;
        for (char c : s) {
            if (result.find(c) == std::string::npos) {
                result += c;
            }
        }
        return result;
    };

    std::string with_dups = "programming";
    std::cout << "   Original: " << with_dups << std::endl;
    std::cout << "   Without duplicates: " << remove_duplicates(with_dups) << std::endl;

    // EXERCISE 7: Reverse Words in String
    std::cout << "\n7. REVERSE WORDS:" << std::endl;
    auto reverse_words = [](const std::string& s) {
        std::vector<std::string> words;
        std::stringstream ss(s);
        std::string word;

        while (ss >> word) {
            words.push_back(word);
        }

        std::reverse(words.begin(), words.end());

        std::string result;
        for (size_t i = 0; i < words.size(); i++) {
            result += words[i];
            if (i < words.size() - 1) result += " ";
        }

        return result;
    };

    std::string sentence = "Hello World from C++";
    std::cout << "   Original: " << sentence << std::endl;
    std::cout << "   Reversed: " << reverse_words(sentence) << std::endl;

    // EXERCISE 8: First Non-Repeating Character
    std::cout << "\n8. FIRST NON-REPEATING CHARACTER:" << std::endl;
    auto first_unique_char = [](const std::string& s) -> char {
        std::map<char, int> freq;

        for (char c : s) {
            freq[c]++;
        }

        for (char c : s) {
            if (freq[c] == 1) {
                return c;
            }
        }

        return '\0';
    };

    std::string test_str = "leetcode";
    char unique = first_unique_char(test_str);
    std::cout << "   String: " << test_str << std::endl;
    std::cout << "   First unique: " << (unique != '\0' ? std::string(1, unique) : "None") << std::endl;

    // EXERCISE 9: String to Integer (atoi)
    std::cout << "\n9. STRING TO INTEGER (custom implementation):" << std::endl;
    auto my_atoi = [](const std::string& s) {
        int result = 0;
        int sign = 1;
        size_t i = 0;

        // Skip whitespace
        while (i < s.length() && s[i] == ' ') i++;

        // Check sign
        if (i < s.length() && (s[i] == '+' || s[i] == '-')) {
            sign = (s[i] == '-') ? -1 : 1;
            i++;
        }

        // Convert digits
        while (i < s.length() && std::isdigit(s[i])) {
            result = result * 10 + (s[i] - '0');
            i++;
        }

        return result * sign;
    };

    std::string num_str = "  -42abc";
    std::cout << "   String: \"" << num_str << "\"" << std::endl;
    std::cout << "   Integer: " << my_atoi(num_str) << std::endl;

    // EXERCISE 10: Longest Palindrome Substring
    std::cout << "\n10. LONGEST PALINDROME SUBSTRING:" << std::endl;
    auto longest_palindrome = [](const std::string& s) {
        if (s.empty()) return std::string("");

        int start = 0, max_len = 1;

        auto expand_around_center = [&](int left, int right) {
            while (left >= 0 && right < (int)s.length() && s[left] == s[right]) {
                int len = right - left + 1;
                if (len > max_len) {
                    start = left;
                    max_len = len;
                }
                left--;
                right++;
            }
        };

        for (int i = 0; i < (int)s.length(); i++) {
            expand_around_center(i, i);      // Odd length palindromes
            expand_around_center(i, i + 1);  // Even length palindromes
        }

        return s.substr(start, max_len);
    };

    std::string palin_test = "babad";
    std::cout << "   String: " << palin_test << std::endl;
    std::cout << "   Longest palindrome: " << longest_palindrome(palin_test) << std::endl;

    // EXERCISE 11: Word Pattern Match
    std::cout << "\n11. WORD PATTERN MATCHING:" << std::endl;
    auto word_pattern = [](const std::string& pattern, const std::string& str) {
        std::vector<std::string> words;
        std::stringstream ss(str);
        std::string word;

        while (ss >> word) {
            words.push_back(word);
        }

        if (pattern.length() != words.size()) return false;

        std::map<char, std::string> char_to_word;
        std::map<std::string, char> word_to_char;

        for (size_t i = 0; i < pattern.length(); i++) {
            char c = pattern[i];
            std::string w = words[i];

            if (char_to_word.count(c) && char_to_word[c] != w) return false;
            if (word_to_char.count(w) && word_to_char[w] != c) return false;

            char_to_word[c] = w;
            word_to_char[w] = c;
        }

        return true;
    };

    std::string pattern = "abba";
    std::string str_test = "dog cat cat dog";
    std::cout << "   Pattern: " << pattern << std::endl;
    std::cout << "   String: " << str_test << std::endl;
    std::cout << "   Matches: " << (word_pattern(pattern, str_test) ? "Yes" : "No") << std::endl;

    // EXERCISE 12: Caesar Cipher
    std::cout << "\n12. CAESAR CIPHER:" << std::endl;
    auto caesar_cipher = [](std::string s, int shift) {
        for (char& c : s) {
            if (std::isalpha(c)) {
                char base = std::isupper(c) ? 'A' : 'a';
                c = base + (c - base + shift) % 26;
            }
        }
        return s;
    };

    std::string message = "Hello World";
    int shift = 3;
    std::string encrypted = caesar_cipher(message, shift);
    std::string decrypted = caesar_cipher(encrypted, 26 - shift);

    std::cout << "   Original: " << message << std::endl;
    std::cout << "   Encrypted (shift " << shift << "): " << encrypted << std::endl;
    std::cout << "   Decrypted: " << decrypted << std::endl;

    // EXERCISE 13: PRACTICE CHALLENGES
    std::cout << "\n13. MORE PRACTICE CHALLENGES:" << std::endl;
    std::cout << "   Try implementing these yourself:" << std::endl;
    std::cout << "   - Count vowels and consonants" << std::endl;
    std::cout << "   - Check if string is a rotation of another" << std::endl;
    std::cout << "   - Find all permutations of a string" << std::endl;
    std::cout << "   - Implement wildcard pattern matching" << std::endl;
    std::cout << "   - Check if two strings are one edit distance apart" << std::endl;
    std::cout << "   - Group anagrams from a list of words" << std::endl;
    std::cout << "   - Implement string multiplication (e.g., \"123\" * \"456\")" << std::endl;
    std::cout << "   - Find the longest substring without repeating characters" << std::endl;

    return 0;
}

/**
 * ADVANCED STRING CONCEPTS COVERED:
 *
 * 1. ANAGRAM DETECTION:
 *    - Sort both strings and compare
 *    - Or use character frequency counting
 *
 * 2. WORD FREQUENCY:
 *    - Use map<string, int> to count occurrences
 *    - Useful for text analysis
 *
 * 3. COMMON PREFIX:
 *    - Compare strings character by character
 *    - Stop when mismatch found
 *
 * 4. PARENTHESES MATCHING:
 *    - Use stack data structure
 *    - Push opening brackets, pop on closing
 *
 * 5. STRING COMPRESSION:
 *    - Run-length encoding
 *    - Count consecutive characters
 *
 * 6. DUPLICATE REMOVAL:
 *    - Track seen characters
 *    - Or use set data structure
 *
 * 7. WORD REVERSAL:
 *    - Split into words
 *    - Reverse vector of words
 *    - Join back
 *
 * 8. UNIQUE CHARACTER FINDING:
 *    - Count character frequencies
 *    - Find first with count of 1
 *
 * 9. STRING PARSING:
 *    - Handle whitespace
 *    - Handle signs
 *    - Convert digit by digit
 *
 * 10. PALINDROME DETECTION:
 *     - Expand around center technique
 *     - Check both odd and even length
 *
 * 11. PATTERN MATCHING:
 *     - Use bidirectional mapping
 *     - Ensure one-to-one correspondence
 *
 * 12. ENCRYPTION:
 *     - Character shifting
 *     - Modular arithmetic
 *     - Preserve case
 *
 * PROBLEM-SOLVING STRATEGIES:
 *
 * 1. TWO-POINTER TECHNIQUE:
 *    - One pointer at start, one at end
 *    - Move towards center
 *    - Useful for palindromes, reversal
 *
 * 2. SLIDING WINDOW:
 *    - Maintain a window of characters
 *    - Slide window through string
 *    - Good for substring problems
 *
 * 3. HASH MAP/SET:
 *    - Track seen characters/substrings
 *    - O(1) lookup time
 *    - Useful for duplicates, anagrams
 *
 * 4. STACK:
 *    - Last-in-first-out
 *    - Useful for matching, parsing
 *    - Expression evaluation
 *
 * 5. DYNAMIC PROGRAMMING:
 *    - Break into subproblems
 *    - Store intermediate results
 *    - Longest common subsequence, etc.
 *
 * COMMON STRING INTERVIEW QUESTIONS:
 * - Reverse string/words
 * - Anagram checking
 * - Palindrome detection
 * - Substring search
 * - String permutations
 * - Character frequency
 * - Pattern matching
 * - String compression
 * - Longest substring without repeating characters
 * - Valid parentheses
 *
 * TIPS FOR STRING PROBLEMS:
 * 1. Clarify requirements (case-sensitive?)
 * 2. Consider edge cases (empty string, single character)
 * 3. Think about time and space complexity
 * 4. Use appropriate data structures
 * 5. Test with multiple examples
 * 6. Consider Unicode vs ASCII
 * 7. Watch for off-by-one errors
 *
 * COMPILE AND RUN:
 * g++ 15_advanced_exercises.cpp -o exercises
 * ./exercises
 *
 * FURTHER LEARNING:
 * - Regular expressions (regex)
 * - String algorithms (KMP, Rabin-Karp)
 * - Trie data structure
 * - String hashing
 * - Suffix arrays and trees
 */
