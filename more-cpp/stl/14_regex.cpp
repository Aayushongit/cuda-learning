/**
 * 14_regex.cpp
 *
 * REGULAR EXPRESSIONS
 * - Pattern matching
 * - Searching
 * - Replacing
 * - Tokenizing
 */

#include <iostream>
#include <regex>
#include <string>
#include <vector>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== REGULAR EXPRESSIONS ===\n";

    separator("BASIC MATCHING");

    // 1. regex_match (entire string must match)
    std::cout << "\n1. REGEX_MATCH:\n";
    std::regex pattern1("\\d+");  // One or more digits

    std::cout << "Match '12345': " << (std::regex_match("12345", pattern1) ? "yes" : "no") << "\n";
    std::cout << "Match 'abc': " << (std::regex_match("abc", pattern1) ? "yes" : "no") << "\n";
    std::cout << "Match '123abc': " << (std::regex_match("123abc", pattern1) ? "yes" : "no") << "\n";

    // 2. regex_search (find pattern anywhere)
    std::cout << "\n2. REGEX_SEARCH:\n";
    std::regex pattern2("\\d+");

    std::cout << "Search in 'abc123def': " << (std::regex_search("abc123def", pattern2) ? "found" : "not found") << "\n";
    std::cout << "Search in 'abcdef': " << (std::regex_search("abcdef", pattern2) ? "found" : "not found") << "\n";

    separator("CAPTURE GROUPS");

    // 3. smatch (capture results)
    std::cout << "\n3. CAPTURE GROUPS:\n";
    std::string text = "Date: 2024-01-15";
    std::regex date_pattern(R"((\d{4})-(\d{2})-(\d{2}))");
    std::smatch match;

    if (std::regex_search(text, match, date_pattern)) {
        std::cout << "Full match: " << match[0] << "\n";
        std::cout << "Year: " << match[1] << "\n";
        std::cout << "Month: " << match[2] << "\n";
        std::cout << "Day: " << match[3] << "\n";
    }

    // 4. Email Validation
    std::cout << "\n4. EMAIL VALIDATION:\n";
    std::regex email_pattern(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");

    std::vector<std::string> emails = {
        "user@example.com",
        "invalid.email",
        "test.user+tag@domain.co.uk"
    };

    for (const auto& email : emails) {
        std::cout << email << ": " << (std::regex_match(email, email_pattern) ? "valid" : "invalid") << "\n";
    }

    separator("SEARCHING");

    // 5. Find All Matches
    std::cout << "\n5. FIND ALL MATCHES:\n";
    std::string document = "Phone: 123-456-7890, Alt: 987-654-3210";
    std::regex phone_pattern(R"(\d{3}-\d{3}-\d{4})");

    auto words_begin = std::sregex_iterator(document.begin(), document.end(), phone_pattern);
    auto words_end = std::sregex_iterator();

    std::cout << "Found " << std::distance(words_begin, words_end) << " phone numbers:\n";
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::cout << "  " << (*i).str() << "\n";
    }

    // 6. Extract Words
    std::cout << "\n6. EXTRACT WORDS:\n";
    std::string sentence = "The quick brown fox jumps over the lazy dog";
    std::regex word_pattern(R"(\b\w+\b)");

    auto word_begin = std::sregex_iterator(sentence.begin(), sentence.end(), word_pattern);
    auto word_end = std::sregex_iterator();

    std::cout << "Words: ";
    for (std::sregex_iterator i = word_begin; i != word_end; ++i) {
        std::cout << (*i).str() << " ";
    }
    std::cout << "\n";

    separator("REPLACING");

    // 7. regex_replace
    std::cout << "\n7. REGEX_REPLACE:\n";
    std::string html = "<p>Hello <b>World</b>!</p>";
    std::regex tag_pattern("<[^>]*>");

    std::string no_tags = std::regex_replace(html, tag_pattern, "");
    std::cout << "Original: " << html << "\n";
    std::cout << "Without tags: " << no_tags << "\n";

    // 8. Replace with Groups
    std::cout << "\n8. REPLACE WITH GROUPS:\n";
    std::string dates = "2024-01-15 and 2024-12-25";
    std::regex date_regex(R"((\d{4})-(\d{2})-(\d{2}))");

    std::string reformatted = std::regex_replace(dates, date_regex, "$2/$3/$1");
    std::cout << "Original: " << dates << "\n";
    std::cout << "Reformatted: " << reformatted << "\n";

    // 9. Censor Words
    std::cout << "\n9. CENSOR WORDS:\n";
    std::string message = "This is bad and terrible";
    std::regex bad_words(R"(\b(bad|terrible)\b)", std::regex::icase);

    std::string censored = std::regex_replace(message, bad_words, "***");
    std::cout << "Original: " << message << "\n";
    std::cout << "Censored: " << censored << "\n";

    separator("TOKENIZING");

    // 10. Split String
    std::cout << "\n10. SPLIT STRING:\n";
    std::string csv = "apple,banana, cherry , date";
    std::regex delimiter(R"(\s*,\s*)");  // Comma with optional spaces

    std::sregex_token_iterator iter(csv.begin(), csv.end(), delimiter, -1);
    std::sregex_token_iterator end;

    std::cout << "Tokens:\n";
    for (; iter != end; ++iter) {
        std::cout << "  '" << *iter << "'\n";
    }

    separator("COMMON PATTERNS");

    // 11. Common Regex Patterns
    std::cout << "\n11. COMMON PATTERNS:\n";

    struct Pattern {
        std::string name;
        std::string regex;
        std::string test_str;
    };

    std::vector<Pattern> patterns = {
        {"US Phone", R"(\(\d{3}\) \d{3}-\d{4})", "(555) 123-4567"},
        {"URL", R"(https?://[^\s]+)", "https://example.com"},
        {"IPv4", R"(\b(?:\d{1,3}\.){3}\d{1,3}\b)", "192.168.1.1"},
        {"Hex Color", R"(#[0-9A-Fa-f]{6})", "#FF5733"},
        {"Time", R"(\d{1,2}:\d{2}(:\d{2})?)", "14:30:00"}
    };

    for (const auto& p : patterns) {
        std::regex r(p.regex);
        bool matches = std::regex_search(p.test_str, r);
        std::cout << p.name << ": " << p.test_str << " -> " << (matches ? "match" : "no match") << "\n";
    }

    separator("REGEX FLAGS");

    // 12. Case Insensitive
    std::cout << "\n12. CASE INSENSITIVE:\n";
    std::regex case_sensitive("hello");
    std::regex case_insensitive("hello", std::regex::icase);

    std::cout << "Case sensitive 'HELLO': " << (std::regex_match("HELLO", case_sensitive) ? "match" : "no match") << "\n";
    std::cout << "Case insensitive 'HELLO': " << (std::regex_match("HELLO", case_insensitive) ? "match" : "no match") << "\n";

    separator("VALIDATION EXAMPLES");

    // 13. Password Validation
    std::cout << "\n13. PASSWORD VALIDATION:\n";
    // At least 8 chars, 1 uppercase, 1 lowercase, 1 digit
    std::regex password_pattern(R"(^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$)");

    std::vector<std::string> passwords = {
        "Pass1234",
        "weakpass",
        "NOLOWER1",
        "NoDigit",
        "Valid1Pass"
    };

    for (const auto& pwd : passwords) {
        std::cout << pwd << ": " << (std::regex_match(pwd, password_pattern) ? "strong" : "weak") << "\n";
    }

    // 14. Extract Numbers
    std::cout << "\n14. EXTRACT ALL NUMBERS:\n";
    std::string mixed = "I have 5 apples, 10 oranges, and 3.5 kg of bananas";
    std::regex number_pattern(R"(\d+\.?\d*)");

    auto num_begin = std::sregex_iterator(mixed.begin(), mixed.end(), number_pattern);
    auto num_end = std::sregex_iterator();

    std::cout << "Numbers found: ";
    for (auto i = num_begin; i != num_end; ++i) {
        std::cout << (*i).str() << " ";
    }
    std::cout << "\n";

    // 15. Greedy vs Non-Greedy
    std::cout << "\n15. GREEDY VS NON-GREEDY:\n";
    std::string html2 = "<div>Content1</div><div>Content2</div>";

    std::regex greedy("<div>.*</div>");
    std::regex non_greedy("<div>.*?</div>");

    std::smatch m1, m2;
    std::regex_search(html2, m1, greedy);
    std::regex_search(html2, m2, non_greedy);

    std::cout << "Greedy match: " << m1[0] << "\n";
    std::cout << "Non-greedy match: " << m2[0] << "\n";

    std::cout << "\n=== END OF REGULAR EXPRESSIONS ===\n";

    return 0;
}
