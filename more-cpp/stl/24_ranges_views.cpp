/**
 * 24_ranges_views.cpp
 *
 * RANGES AND VIEWS (C++20)
 * - Range concepts
 * - Range algorithms
 * - Range views (lazy evaluation)
 * - Range adapters
 * - Range pipelines
 *
 * Note: Requires C++20 compiler support
 */

#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>
#include <string>

namespace ranges = std::ranges;
namespace views = std::views;

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== RANGES AND VIEWS (C++20) ===\n";

    separator("BASIC RANGES");

    // 1. Range Algorithms
    std::cout << "\n1. RANGE ALGORITHMS:\n";
    std::vector<int> nums = {5, 2, 8, 1, 9, 3};

    // Sort entire range
    ranges::sort(nums);

    std::cout << "Sorted: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << "\n";

    // Find
    auto it = ranges::find(nums, 8);
    if (it != nums.end()) {
        std::cout << "Found: " << *it << "\n";
    }

    // Count_if
    int even_count = ranges::count_if(nums, [](int n) { return n % 2 == 0; });
    std::cout << "Even numbers: " << even_count << "\n";

    // 2. Projections
    std::cout << "\n2. PROJECTIONS:\n";
    struct Person {
        std::string name;
        int age;
    };

    std::vector<Person> people = {
        {"Alice", 30},
        {"Bob", 25},
        {"Charlie", 35}
    };

    // Sort by age using projection
    ranges::sort(people, {}, &Person::age);

    std::cout << "Sorted by age:\n";
    for (const auto& p : people) {
        std::cout << "  " << p.name << " (" << p.age << ")\n";
    }

    separator("VIEWS");

    // 3. filter view
    std::cout << "\n3. FILTER VIEW:\n";
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto evens = numbers | views::filter([](int n) { return n % 2 == 0; });

    std::cout << "Even numbers: ";
    for (int n : evens) std::cout << n << " ";
    std::cout << "\n";

    // 4. transform view
    std::cout << "\n4. TRANSFORM VIEW:\n";
    auto squares = numbers | views::transform([](int n) { return n * n; });

    std::cout << "Squares: ";
    for (int n : squares) std::cout << n << " ";
    std::cout << "\n";

    // 5. take view
    std::cout << "\n5. TAKE VIEW:\n";
    auto first_five = numbers | views::take(5);

    std::cout << "First 5: ";
    for (int n : first_five) std::cout << n << " ";
    std::cout << "\n";

    // 6. drop view
    std::cout << "\n6. DROP VIEW:\n";
    auto skip_three = numbers | views::drop(3);

    std::cout << "After dropping 3: ";
    for (int n : skip_three) std::cout << n << " ";
    std::cout << "\n";

    // 7. reverse view
    std::cout << "\n7. REVERSE VIEW:\n";
    auto reversed = numbers | views::reverse;

    std::cout << "Reversed: ";
    for (int n : reversed) std::cout << n << " ";
    std::cout << "\n";

    separator("VIEW PIPELINES");

    // 8. Composing Views
    std::cout << "\n8. COMPOSING VIEWS:\n";
    auto result = numbers
        | views::filter([](int n) { return n % 2 == 0; })
        | views::transform([](int n) { return n * n; })
        | views::take(3);

    std::cout << "First 3 squares of even numbers: ";
    for (int n : result) std::cout << n << " ";
    std::cout << "\n";

    // 9. Complex Pipeline
    std::cout << "\n9. COMPLEX PIPELINE:\n";
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto pipeline = data
        | views::drop(2)                              // Skip first 2
        | views::filter([](int n) { return n > 3; })  // Only > 3
        | views::transform([](int n) { return n * 2; }) // Double
        | views::take(4);                             // Take 4

    std::cout << "Pipeline result: ";
    for (int n : pipeline) std::cout << n << " ";
    std::cout << "\n";

    separator("LAZY EVALUATION");

    // 10. Lazy Evaluation Demo
    std::cout << "\n10. LAZY EVALUATION:\n";
    std::cout << "Views don't compute until accessed:\n";

    auto lazy_view = numbers
        | views::transform([](int n) {
            std::cout << "  Transform " << n << "\n";
            return n * 2;
        })
        | views::take(3);

    std::cout << "View created (not evaluated yet)\n";
    std::cout << "Now iterating:\n";
    for (int n : lazy_view) {
        std::cout << "  Got: " << n << "\n";
    }

    separator("MORE VIEWS");

    // 11. split view
    std::cout << "\n11. SPLIT VIEW:\n";
    std::string text = "hello world from cpp";
    auto words = text | views::split(' ');

    std::cout << "Words: ";
    for (const auto& word : words) {
        for (char c : word) std::cout << c;
        std::cout << " | ";
    }
    std::cout << "\n";

    // 12. iota view (infinite range)
    std::cout << "\n12. IOTA VIEW:\n";
    auto infinite = views::iota(1);  // 1, 2, 3, ...
    auto first_ten = infinite | views::take(10);

    std::cout << "First 10 natural numbers: ";
    for (int n : first_ten) std::cout << n << " ";
    std::cout << "\n";

    // 13. keys and values views
    std::cout << "\n13. KEYS AND VALUES VIEWS:\n";
    std::vector<std::pair<std::string, int>> pairs = {
        {"apple", 1},
        {"banana", 2},
        {"cherry", 3}
    };

    std::cout << "Keys: ";
    for (const auto& key : pairs | views::keys) {
        std::cout << key << " ";
    }
    std::cout << "\n";

    std::cout << "Values: ";
    for (const auto& val : pairs | views::values) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // 14. join view
    std::cout << "\n14. JOIN VIEW:\n";
    std::vector<std::vector<int>> nested = {{1, 2}, {3, 4}, {5, 6}};
    auto flattened = nested | views::join;

    std::cout << "Flattened: ";
    for (int n : flattened) std::cout << n << " ";
    std::cout << "\n";

    separator("PRACTICAL EXAMPLES");

    // 15. Process CSV-like Data
    std::cout << "\n15. PROCESS DATA PIPELINE:\n";
    std::vector<int> sales = {100, 50, 200, 75, 150, 300, 25, 175};

    auto top_sales = sales
        | views::filter([](int n) { return n >= 100; })
        | views::transform([](int n) { return n * 1.1; })  // 10% bonus
        | views::take(3);

    std::cout << "Top 3 sales with bonus: ";
    for (double n : top_sales) std::cout << n << " ";
    std::cout << "\n";

    // 16. Fibonacci with Views
    std::cout << "\n16. FIBONACCI (lazy):\n";
    std::vector<long long> fib = {0, 1};
    // Generate first 15 Fibonacci numbers
    while (fib.size() < 15) {
        fib.push_back(fib[fib.size() - 1] + fib[fib.size() - 2]);
    }

    auto first_10_fib = fib | views::take(10);
    std::cout << "First 10 Fibonacci: ";
    for (long long n : first_10_fib) std::cout << n << " ";
    std::cout << "\n";

    separator("ADVANTAGES OF RANGES");

    std::cout << "\n1. COMPOSABILITY:\n";
    std::cout << "   - Chain operations with | operator\n";
    std::cout << "   - More readable than nested function calls\n";

    std::cout << "\n2. LAZY EVALUATION:\n";
    std::cout << "   - Views don't compute until accessed\n";
    std::cout << "   - No intermediate containers\n";
    std::cout << "   - Memory efficient\n";

    std::cout << "\n3. NO ITERATORS:\n";
    std::cout << "   - Work directly with ranges\n";
    std::cout << "   - Less error-prone\n";
    std::cout << "   - More expressive\n";

    std::cout << "\n4. PROJECTIONS:\n";
    std::cout << "   - Sort/search by member directly\n";
    std::cout << "   - No need for custom comparators\n";

    separator("RANGES VS TRADITIONAL");

    std::cout << "\n17. COMPARISON:\n\n";

    std::cout << "Traditional:\n";
    std::cout << "  std::vector<int> temp;\n";
    std::cout << "  std::copy_if(vec.begin(), vec.end(), back_inserter(temp), pred);\n";
    std::cout << "  std::transform(temp.begin(), temp.end(), temp.begin(), func);\n";
    std::cout << "  std::vector<int> result(temp.begin(), temp.begin() + 5);\n\n";

    std::cout << "Ranges:\n";
    std::cout << "  auto result = vec\n";
    std::cout << "    | views::filter(pred)\n";
    std::cout << "    | views::transform(func)\n";
    std::cout << "    | views::take(5);\n";

    separator("BEST PRACTICES");

    std::cout << "\n1. Use ranges for cleaner, more expressive code\n";
    std::cout << "2. Leverage lazy evaluation for performance\n";
    std::cout << "3. Compose views for complex transformations\n";
    std::cout << "4. Use projections instead of custom comparators\n";
    std::cout << "5. Views are lightweight - cheap to copy\n";
    std::cout << "6. Be aware of lifetime issues with views\n";
    std::cout << "7. Materialize views when needed (ranges::to in C++23)\n";
    std::cout << "8. Ranges represent a modern C++ paradigm\n";

    std::cout << "\n=== END OF RANGES AND VIEWS ===\n";

    return 0;
}
