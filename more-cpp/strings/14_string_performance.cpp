/**
 * 14_string_performance.cpp
 *
 * String Performance and Best Practices
 *
 * LEARNING OBJECTIVES:
 * - Understand string performance characteristics
 * - Learn optimization techniques
 * - Avoid common performance pitfalls
 * - Memory management with strings
 * - Best practices for efficient string handling
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>  // For timing

int main() {
    std::cout << "=== STRING PERFORMANCE AND BEST PRACTICES ===" << std::endl;

    // 1. STRING CONCATENATION - GOOD vs BAD
    std::cout << "\n1. STRING CONCATENATION:" << std::endl;

    // BAD: Multiple concatenations create temporary strings
    std::cout << "   BAD approach (creates many temporaries):" << std::endl;
    std::string bad = "";
    for (int i = 0; i < 5; i++) {
        bad = bad + "x";  // Creates new string each time!
    }
    std::cout << "   Result: " << bad << std::endl;

    // GOOD: Use += or append
    std::cout << "   GOOD approach (modifies in place):" << std::endl;
    std::string good = "";
    for (int i = 0; i < 5; i++) {
        good += "x";  // More efficient
    }
    std::cout << "   Result: " << good << std::endl;

    // 2. RESERVE CAPACITY
    std::cout << "\n2. RESERVE CAPACITY:" << std::endl;

    // Without reserve
    std::string no_reserve;
    std::cout << "   Without reserve:" << std::endl;
    std::cout << "   Initial capacity: " << no_reserve.capacity() << std::endl;
    for (int i = 0; i < 100; i++) {
        no_reserve += 'x';
    }
    std::cout << "   Final capacity: " << no_reserve.capacity() << std::endl;

    // With reserve
    std::string with_reserve;
    with_reserve.reserve(100);
    std::cout << "\n   With reserve(100):" << std::endl;
    std::cout << "   Initial capacity: " << with_reserve.capacity() << std::endl;
    for (int i = 0; i < 100; i++) {
        with_reserve += 'x';
    }
    std::cout << "   Final capacity: " << with_reserve.capacity() << std::endl;
    std::cout << "   (Fewer reallocations = better performance)" << std::endl;

    // 3. PASS BY REFERENCE
    std::cout << "\n3. PASS BY REFERENCE:" << std::endl;

    // BAD: Pass by value (copies the string)
    auto bad_function = [](std::string str) {
        return str.length();
    };

    // GOOD: Pass by const reference (no copy)
    auto good_function = [](const std::string& str) {
        return str.length();
    };

    std::string test = "Hello World";
    std::cout << "   Both functions work correctly:" << std::endl;
    std::cout << "   Pass by value: " << bad_function(test) << std::endl;
    std::cout << "   Pass by const ref: " << good_function(test) << std::endl;
    std::cout << "   (But const ref is much faster for large strings)" << std::endl;

    // 4. AVOID UNNECESSARY COPIES
    std::cout << "\n4. AVOID UNNECESSARY COPIES:" << std::endl;

    // BAD: Creates copy
    std::string original = "Original String";
    std::string copy = original;  // Full copy
    std::cout << "   Copy made: " << copy << std::endl;

    // BETTER: Use reference when not modifying
    const std::string& ref = original;  // No copy
    std::cout << "   Reference: " << ref << std::endl;

    // BEST: Use move when transferring ownership
    std::string moved = std::move(original);  // Transfer, no copy
    std::cout << "   Moved: " << moved << std::endl;
    std::cout << "   Original after move: '" << original << "'" << std::endl;

    // 5. STRING BUILDER PATTERN
    std::cout << "\n5. STRING BUILDER PATTERN:" << std::endl;

    auto build_string = []() {
        std::string result;
        result.reserve(50);  // Reserve space upfront

        result += "Building ";
        result += "a ";
        result += "string ";
        result += "efficiently!";

        return result;
    };

    std::cout << "   Result: " << build_string() << std::endl;

    // 6. SMALL STRING OPTIMIZATION (SSO)
    std::cout << "\n6. SMALL STRING OPTIMIZATION:" << std::endl;
    std::cout << "   Many compilers optimize small strings (<= ~15 chars)" << std::endl;
    std::cout << "   These are stored directly in the string object (no heap allocation)" << std::endl;

    std::string small = "Small";
    std::string large = "This is a much larger string that likely needs heap allocation";

    std::cout << "   Small string: " << small << " (capacity: " << small.capacity() << ")" << std::endl;
    std::cout << "   Large string length: " << large.length() << " (capacity: " << large.capacity() << ")" << std::endl;

    // 7. CLEAR vs EMPTY STRING
    std::cout << "\n7. CLEAR vs ASSIGNMENT:" << std::endl;

    std::string str = "Some content here";
    std::cout << "   Original capacity: " << str.capacity() << std::endl;

    str.clear();  // Clears content, keeps capacity
    std::cout << "   After clear() - capacity: " << str.capacity() << std::endl;

    str = "";  // Also clears, but may deallocate
    std::cout << "   After = \"\" - capacity: " << str.capacity() << std::endl;

    // 8. SHRINK_TO_FIT
    std::cout << "\n8. SHRINK_TO_FIT:" << std::endl;

    std::string shrinkable;
    shrinkable.reserve(1000);
    shrinkable = "Short";

    std::cout << "   After reserve(1000) then short string:" << std::endl;
    std::cout << "   Length: " << shrinkable.length() << ", Capacity: " << shrinkable.capacity() << std::endl;

    shrinkable.shrink_to_fit();  // Reduce capacity to fit content
    std::cout << "   After shrink_to_fit():" << std::endl;
    std::cout << "   Length: " << shrinkable.length() << ", Capacity: " << shrinkable.capacity() << std::endl;

    // 9. C_STR() COST
    std::cout << "\n9. C_STR() CONSIDERATIONS:" << std::endl;
    std::string cpp_str = "Hello";

    std::cout << "   c_str() is O(1) but the pointer can become invalid" << std::endl;
    const char* ptr = cpp_str.c_str();
    std::cout << "   Pointer: " << ptr << std::endl;

    cpp_str += " World";  // String modified
    std::cout << "   After modification, old pointer may be invalid!" << std::endl;
    // Don't use 'ptr' here - it might be invalid

    // 10. SUBSTRING CREATION
    std::cout << "\n10. SUBSTRING OPERATIONS:" << std::endl;

    std::string base = "Hello World Programming";
    std::cout << "   Creating substring copies data:" << std::endl;

    std::string sub = base.substr(6, 5);  // Creates new string
    std::cout << "   Substring: " << sub << std::endl;
    std::cout << "   (C++17 string_view can avoid this copy)" << std::endl;

    // 11. COMPARISON OPTIMIZATION
    std::cout << "\n11. COMPARISON TIPS:" << std::endl;

    std::string s1 = "Test";
    std::string s2 = "Test";

    // Check length first for inequality
    if (s1.length() != s2.length()) {
        std::cout << "   Different (length check is fast)" << std::endl;
    } else {
        if (s1 == s2) {
            std::cout << "   Equal (full comparison needed)" << std::endl;
        }
    }

    // 12. MEASURING PERFORMANCE
    std::cout << "\n12. PERFORMANCE MEASUREMENT EXAMPLE:" << std::endl;

    // Bad approach
    auto start_bad = std::chrono::high_resolution_clock::now();
    std::string result_bad;
    for (int i = 0; i < 1000; i++) {
        result_bad = result_bad + "x";
    }
    auto end_bad = std::chrono::high_resolution_clock::now();
    auto duration_bad = std::chrono::duration_cast<std::chrono::microseconds>(end_bad - start_bad);

    // Good approach
    auto start_good = std::chrono::high_resolution_clock::now();
    std::string result_good;
    result_good.reserve(1000);
    for (int i = 0; i < 1000; i++) {
        result_good += "x";
    }
    auto end_good = std::chrono::high_resolution_clock::now();
    auto duration_good = std::chrono::duration_cast<std::chrono::microseconds>(end_good - start_good);

    std::cout << "   Bad approach (operator+): " << duration_bad.count() << " μs" << std::endl;
    std::cout << "   Good approach (+=): " << duration_good.count() << " μs" << std::endl;

    // 13. MEMORY MANAGEMENT
    std::cout << "\n13. MEMORY AWARENESS:" << std::endl;
    std::cout << "   - Strings allocate dynamically (except SSO)" << std::endl;
    std::cout << "   - Capacity >= length always" << std::endl;
    std::cout << "   - Growth usually doubles capacity" << std::endl;
    std::cout << "   - clear() keeps capacity, shrink_to_fit() releases" << std::endl;

    // 14. BEST PRACTICES SUMMARY
    std::cout << "\n14. BEST PRACTICES SUMMARY:" << std::endl;
    std::cout << "   ✓ Use += instead of + for concatenation" << std::endl;
    std::cout << "   ✓ Reserve capacity when size is known" << std::endl;
    std::cout << "   ✓ Pass by const reference, not by value" << std::endl;
    std::cout << "   ✓ Use move semantics when transferring" << std::endl;
    std::cout << "   ✓ Avoid unnecessary substr() calls" << std::endl;
    std::cout << "   ✓ Use string_view (C++17) for read-only substrings" << std::endl;
    std::cout << "   ✓ Be aware of Small String Optimization" << std::endl;
    std::cout << "   ✓ Reuse strings instead of creating new ones" << std::endl;
    std::cout << "   ✓ Check length before full comparison" << std::endl;
    std::cout << "   ✓ Profile before optimizing" << std::endl;

    return 0;
}

/**
 * STRING PERFORMANCE GUIDE:
 *
 * OPERATION COMPLEXITY:
 * - Access []:        O(1)
 * - Length:           O(1)
 * - Append +=:        O(1) amortized
 * - Insert:           O(n)
 * - Find:             O(n*m)
 * - Substring:        O(n) - creates copy
 * - Compare:          O(n)
 *
 * MEMORY CHARACTERISTICS:
 * 1. Dynamic allocation (heap) for most strings
 * 2. Small String Optimization (SSO):
 *    - Strings <= ~15 chars stored in object
 *    - No heap allocation needed
 *    - Implementation-specific
 *
 * 3. Capacity vs Size:
 *    - Size: actual string length
 *    - Capacity: allocated memory
 *    - Growth: typically doubles capacity
 *
 * PERFORMANCE DOS AND DON'TS:
 *
 * ✓ DO:
 * 1. Use += for appending
 *    str += "text";
 *
 * 2. Reserve capacity
 *    str.reserve(expected_size);
 *
 * 3. Pass by const reference
 *    void func(const string& s);
 *
 * 4. Use move when transferring
 *    string s2 = std::move(s1);
 *
 * 5. Reuse strings
 *    str.clear();  // Reuse for next operation
 *
 * ✗ DON'T:
 * 1. Use + in loops
 *    for (...) str = str + "x";  // BAD!
 *
 * 2. Pass by value unnecessarily
 *    void func(string s);  // Copies!
 *
 * 3. Create many temporaries
 *    str = s1 + s2 + s3 + s4;  // Multiple temps
 *
 * 4. Ignore capacity
 *    // Will reallocate multiple times
 *    for (int i = 0; i < 10000; i++) str += "x";
 *
 * OPTIMIZATION STRATEGIES:
 *
 * 1. BUILDER PATTERN:
 *    string result;
 *    result.reserve(estimated_size);
 *    result += part1;
 *    result += part2;
 *    return result;
 *
 * 2. MOVE SEMANTICS (C++11):
 *    vector<string> v;
 *    string s = "data";
 *    v.push_back(std::move(s));  // No copy
 *
 * 3. STRING_VIEW (C++17):
 *    string_view sv = str;  // No copy, just view
 *    // Good for passing substrings
 *
 * 4. RVALUE REFERENCES:
 *    void func(string&& s) {
 *        // Can take ownership without copy
 *    }
 *
 * MEASURING PERFORMANCE:
 *
 * #include <chrono>
 *
 * auto start = chrono::high_resolution_clock::now();
 * // ... operation ...
 * auto end = chrono::high_resolution_clock::now();
 * auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
 * cout << duration.count() << " μs" << endl;
 *
 * COMMON PITFALLS:
 *
 * 1. Repeated concatenation:
 *    BAD:  for (...) result = result + s;
 *    GOOD: for (...) result += s;
 *
 * 2. Copying in function calls:
 *    BAD:  void func(string s);
 *    GOOD: void func(const string& s);
 *
 * 3. Unnecessary substring creation:
 *    BAD:  if (str.substr(0, 5) == "Hello")
 *    GOOD: if (str.compare(0, 5, "Hello") == 0)
 *
 * 4. Not reserving capacity:
 *    BAD:  string s; for (...) s += "x";
 *    GOOD: string s; s.reserve(1000); for (...) s += "x";
 *
 * WHEN TO OPTIMIZE:
 * - Profile first!
 * - Optimize hot paths only
 * - Large strings or many operations
 * - Performance-critical code
 * - Don't sacrifice readability for micro-optimizations
 *
 * COMPILE AND RUN:
 * g++ 14_string_performance.cpp -o performance -O2
 * ./performance
 *
 * Note: Use -O2 or -O3 for realistic performance testing
 */
