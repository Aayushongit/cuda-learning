/**
 * 19_tuple_pair.cpp
 *
 * TUPLE AND PAIR
 * - std::pair: Two elements
 * - std::tuple: Fixed-size collection of heterogeneous values
 * - Structured bindings (C++17)
 */

#include <iostream>
#include <tuple>
#include <utility>
#include <string>
#include <map>
#include <vector>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== TUPLE AND PAIR ===\n";

    separator("STD::PAIR");

    // 1. Pair Basics
    std::cout << "\n1. PAIR BASICS:\n";
    std::pair<int, std::string> p1(1, "one");
    std::pair<int, std::string> p2 = {2, "two"};
    auto p3 = std::make_pair(3, "three");

    std::cout << "p1: (" << p1.first << ", " << p1.second << ")\n";
    std::cout << "p2: (" << p2.first << ", " << p2.second << ")\n";
    std::cout << "p3: (" << p3.first << ", " << p3.second << ")\n";

    // 2. Pair Operations
    std::cout << "\n2. PAIR OPERATIONS:\n";
    p1.first = 10;
    p1.second = "ten";
    std::cout << "Modified p1: (" << p1.first << ", " << p1.second << ")\n";

    // Swap
    std::pair<int, int> a = {1, 2};
    std::pair<int, int> b = {3, 4};
    a.swap(b);
    std::cout << "After swap, a: (" << a.first << ", " << a.second << ")\n";

    // 3. Pair Comparison
    std::cout << "\n3. PAIR COMPARISON:\n";
    std::pair<int, int> p_a = {1, 2};
    std::pair<int, int> p_b = {1, 3};
    std::pair<int, int> p_c = {1, 2};

    std::cout << "p_a == p_c: " << (p_a == p_c) << "\n";
    std::cout << "p_a < p_b: " << (p_a < p_b) << "\n";

    // 4. Pair with Map
    std::cout << "\n4. PAIR WITH MAP:\n";
    std::map<std::string, int> ages;
    auto result = ages.insert(std::make_pair("Alice", 30));

    if (result.second) {
        std::cout << "Inserted: " << result.first->first << " -> " << result.first->second << "\n";
    }

    // 5. Structured Bindings (C++17)
    std::cout << "\n5. STRUCTURED BINDINGS WITH PAIR:\n";
    auto [key, value] = std::make_pair("temperature", 25);
    std::cout << "Key: " << key << ", Value: " << value << "\n";

    for (const auto& [name, age] : ages) {
        std::cout << name << " is " << age << " years old\n";
    }

    separator("STD::TUPLE");

    // 6. Tuple Basics
    std::cout << "\n6. TUPLE BASICS:\n";
    std::tuple<int, double, std::string> t1(1, 3.14, "pi");
    auto t2 = std::make_tuple(2, 2.71, "e");

    std::cout << "t1: (" << std::get<0>(t1) << ", "
              << std::get<1>(t1) << ", "
              << std::get<2>(t1) << ")\n";

    // 7. Tuple Access
    std::cout << "\n7. TUPLE ACCESS:\n";
    auto t = std::make_tuple(100, std::string("text"), 3.14);

    std::cout << "By index - 0: " << std::get<0>(t) << "\n";
    std::cout << "By type - string: " << std::get<std::string>(t) << "\n";

    // Modify
    std::get<0>(t) = 200;
    std::cout << "After modification: " << std::get<0>(t) << "\n";

    // 8. tuple_size and tuple_element
    std::cout << "\n8. TUPLE SIZE AND TYPE:\n";
    std::tuple<int, double, char, std::string> big_tuple;

    std::cout << "Tuple size: " << std::tuple_size<decltype(big_tuple)>::value << "\n";
    std::cout << "Element 1 type is double: "
              << std::is_same_v<std::tuple_element<1, decltype(big_tuple)>::type, double> << "\n";

    // 9. Structured Bindings
    std::cout << "\n9. STRUCTURED BINDINGS WITH TUPLE:\n";
    auto [id, name, score] = std::make_tuple(1, "Alice", 95.5);
    std::cout << "ID: " << id << ", Name: " << name << ", Score: " << score << "\n";

    // 10. Tuple as Return Type
    std::cout << "\n10. TUPLE AS RETURN TYPE:\n";
    auto get_stats = []() {
        return std::make_tuple("Stats", 42, 3.14, true);
    };

    auto [label, count, average, success] = get_stats();
    std::cout << label << ": count=" << count << ", avg=" << average
              << ", success=" << success << "\n";

    // 11. Tuple Concatenation
    std::cout << "\n11. TUPLE CONCATENATION:\n";
    auto t_a = std::make_tuple(1, 2);
    auto t_b = std::make_tuple("a", "b");
    auto combined = std::tuple_cat(t_a, t_b);

    std::cout << "Combined tuple size: " << std::tuple_size<decltype(combined)>::value << "\n";
    std::cout << "Elements: " << std::get<0>(combined) << ", "
              << std::get<1>(combined) << ", "
              << std::get<2>(combined) << ", "
              << std::get<3>(combined) << "\n";

    // 12. Tuple Comparison
    std::cout << "\n12. TUPLE COMPARISON:\n";
    auto tup1 = std::make_tuple(1, 2, 3);
    auto tup2 = std::make_tuple(1, 2, 4);
    auto tup3 = std::make_tuple(1, 2, 3);

    std::cout << "tup1 == tup3: " << (tup1 == tup3) << "\n";
    std::cout << "tup1 < tup2: " << (tup1 < tup2) << "\n";

    // 13. Tuple with References
    std::cout << "\n13. TUPLE WITH REFERENCES:\n";
    int x = 10;
    int y = 20;
    auto ref_tuple = std::tie(x, y);  // Creates tuple of references

    std::get<0>(ref_tuple) = 100;
    std::cout << "After modifying tuple, x = " << x << "\n";

    // Swap using tie
    std::tie(x, y) = std::make_tuple(y, x);
    std::cout << "After swap via tie: x=" << x << ", y=" << y << "\n";

    // 14. Ignore Elements
    std::cout << "\n14. IGNORE ELEMENTS:\n";
    auto data = std::make_tuple(1, 2, 3, 4);
    int first, last;
    std::tie(first, std::ignore, std::ignore, last) = data;
    std::cout << "First: " << first << ", Last: " << last << "\n";

    // 15. Apply Function to Tuple (C++17)
    std::cout << "\n15. APPLY FUNCTION TO TUPLE:\n";
    auto sum = [](int a, int b, int c) {
        return a + b + c;
    };

    auto numbers = std::make_tuple(10, 20, 30);
    int sum_result = std::apply(sum, numbers);
    std::cout << "Sum via apply: " << sum_result << "\n";

    // 16. Tuple as Hash Key
    std::cout << "\n16. TUPLE AS MAP KEY:\n";
    std::map<std::tuple<int, int>, std::string> coordinates;
    coordinates[{0, 0}] = "origin";
    coordinates[{1, 1}] = "diagonal";

    std::cout << "At (0,0): " << coordinates[{0, 0}] << "\n";
    std::cout << "At (1,1): " << coordinates[{1, 1}] << "\n";

    // 17. Practical: Multiple Return Values
    std::cout << "\n17. PRACTICAL: MULTIPLE RETURN VALUES:\n";
    auto divide = [](int a, int b) -> std::tuple<int, int, bool> {
        if (b == 0) return {0, 0, false};
        return {a / b, a % b, true};
    };

    auto [quotient, remainder, success2] = divide(17, 5);
    if (success2) {
        std::cout << "17 / 5 = " << quotient << " remainder " << remainder << "\n";
    }

    // 18. Tuple Unpacking in Loop
    std::cout << "\n18. TUPLE IN CONTAINERS:\n";
    std::vector<std::tuple<std::string, int, double>> students = {
        {"Alice", 20, 3.8},
        {"Bob", 21, 3.5},
        {"Charlie", 19, 3.9}
    };

    for (const auto& [name, age, gpa] : students) {
        std::cout << name << " (age " << age << "): GPA " << gpa << "\n";
    }

    separator("USE CASES");

    std::cout << "\nUse PAIR when:\n";
    std::cout << "- Need exactly two related values\n";
    std::cout << "- Map insert return type\n";
    std::cout << "- Simple key-value associations\n";

    std::cout << "\nUse TUPLE when:\n";
    std::cout << "- Multiple return values from function\n";
    std::cout << "- Fixed-size heterogeneous collections\n";
    std::cout << "- Temporary grouping of values\n";
    std::cout << "- Alternative to small structs\n";

    std::cout << "\nConsider STRUCT when:\n";
    std::cout << "- Named members improve readability\n";
    std::cout << "- Type will be used throughout codebase\n";
    std::cout << "- Need member functions\n";

    std::cout << "\n=== END OF TUPLE AND PAIR ===\n";

    return 0;
}
