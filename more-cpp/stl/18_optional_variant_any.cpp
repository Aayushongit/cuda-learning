/**
 * 18_optional_variant_any.cpp
 *
 * MODERN C++ UTILITY TYPES (C++17)
 * - std::optional: May or may not contain a value
 * - std::variant: Type-safe union
 * - std::any: Type-safe container for any type
 */

#include <iostream>
#include <optional>
#include <variant>
#include <any>
#include <string>
#include <vector>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== OPTIONAL, VARIANT, ANY ===\n";

    separator("STD::OPTIONAL");

    // 1. Basic optional
    std::cout << "\n1. OPTIONAL BASICS:\n";
    std::optional<int> opt1 = 42;
    std::optional<int> opt2 = std::nullopt;  // Empty
    std::optional<int> opt3;                  // Empty

    std::cout << "opt1 has_value: " << opt1.has_value() << "\n";
    std::cout << "opt2 has_value: " << opt2.has_value() << "\n";

    if (opt1) {
        std::cout << "opt1 value: " << *opt1 << "\n";
        std::cout << "opt1 value (via value()): " << opt1.value() << "\n";
    }

    // 2. value_or
    std::cout << "\n2. VALUE_OR:\n";
    std::optional<int> empty_opt;
    std::cout << "Empty value_or(10): " << empty_opt.value_or(10) << "\n";

    std::optional<int> full_opt = 5;
    std::cout << "Full value_or(10): " << full_opt.value_or(10) << "\n";

    // 3. Optional as Return Type
    std::cout << "\n3. OPTIONAL AS RETURN TYPE:\n";
    auto find_in_vector = [](const std::vector<int>& vec, int target) -> std::optional<size_t> {
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] == target) return i;
        }
        return std::nullopt;
    };

    std::vector<int> nums = {10, 20, 30, 40};
    if (auto pos = find_in_vector(nums, 30)) {
        std::cout << "Found at index: " << *pos << "\n";
    } else {
        std::cout << "Not found\n";
    }

    // 4. Emplace and Reset
    std::cout << "\n4. EMPLACE AND RESET:\n";
    std::optional<std::string> opt_str;
    opt_str.emplace("Hello");  // Construct in-place
    std::cout << "After emplace: " << *opt_str << "\n";

    opt_str.reset();  // Clear
    std::cout << "After reset, has_value: " << opt_str.has_value() << "\n";

    // 5. Optional with Custom Types
    std::cout << "\n5. OPTIONAL WITH CUSTOM TYPES:\n";
    struct Point { int x, y; };
    std::optional<Point> opt_point = Point{10, 20};

    if (opt_point) {
        std::cout << "Point: (" << opt_point->x << ", " << opt_point->y << ")\n";
    }

    separator("STD::VARIANT");

    // 6. Variant Basics
    std::cout << "\n6. VARIANT BASICS:\n";
    std::variant<int, double, std::string> var;

    var = 42;
    std::cout << "Holds int: " << std::get<int>(var) << "\n";

    var = 3.14;
    std::cout << "Holds double: " << std::get<double>(var) << "\n";

    var = std::string("Hello");
    std::cout << "Holds string: " << std::get<std::string>(var) << "\n";

    // 7. index() and holds_alternative
    std::cout << "\n7. VARIANT INDEX AND TYPE CHECK:\n";
    std::cout << "Current index: " << var.index() << "\n";  // 2 (string)
    std::cout << "Holds int: " << std::holds_alternative<int>(var) << "\n";
    std::cout << "Holds string: " << std::holds_alternative<std::string>(var) << "\n";

    // 8. get_if (safe access)
    std::cout << "\n8. GET_IF (SAFE ACCESS):\n";
    std::variant<int, std::string> v = 100;

    if (auto p_int = std::get_if<int>(&v)) {
        std::cout << "Contains int: " << *p_int << "\n";
    }

    if (auto p_str = std::get_if<std::string>(&v)) {
        std::cout << "Contains string: " << *p_str << "\n";
    } else {
        std::cout << "Does not contain string\n";
    }

    // 9. Visitor Pattern
    std::cout << "\n9. VISITOR PATTERN:\n";
    std::variant<int, double, std::string> multi = 3.14;

    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int>) {
            std::cout << "Visiting int: " << arg << "\n";
        } else if constexpr (std::is_same_v<T, double>) {
            std::cout << "Visiting double: " << arg << "\n";
        } else if constexpr (std::is_same_v<T, std::string>) {
            std::cout << "Visiting string: " << arg << "\n";
        }
    }, multi);

    // 10. Variant for State Machines
    std::cout << "\n10. VARIANT FOR STATE MACHINE:\n";
    struct Idle {};
    struct Running { int progress; };
    struct Done { std::string result; };

    using State = std::variant<Idle, Running, Done>;

    State state = Running{50};

    std::visit([](auto&& s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, Idle>) {
            std::cout << "State: Idle\n";
        } else if constexpr (std::is_same_v<T, Running>) {
            std::cout << "State: Running at " << s.progress << "%\n";
        } else if constexpr (std::is_same_v<T, Done>) {
            std::cout << "State: Done with result: " << s.result << "\n";
        }
    }, state);

    separator("STD::ANY");

    // 11. Any Basics
    std::cout << "\n11. ANY BASICS:\n";
    std::any a1 = 42;
    std::any a2 = 3.14;
    std::any a3 = std::string("Hello");
    std::any a4;  // Empty

    std::cout << "a4 has_value: " << a4.has_value() << "\n";
    std::cout << "a1 type: " << a1.type().name() << "\n";

    // 12. any_cast
    std::cout << "\n12. ANY_CAST:\n";
    try {
        int value = std::any_cast<int>(a1);
        std::cout << "a1 as int: " << value << "\n";

        // Wrong type throws
        double wrong = std::any_cast<double>(a1);  // Throws!
    } catch (const std::bad_any_cast& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }

    // 13. Safe any_cast with pointer
    std::cout << "\n13. SAFE ANY_CAST:\n";
    if (auto p = std::any_cast<int>(&a1)) {
        std::cout << "a1 contains int: " << *p << "\n";
    }

    if (auto p = std::any_cast<double>(&a1)) {
        std::cout << "a1 contains double: " << *p << "\n";
    } else {
        std::cout << "a1 does not contain double\n";
    }

    // 14. Emplace and Reset
    std::cout << "\n14. ANY EMPLACE AND RESET:\n";
    std::any container;
    container.emplace<std::vector<int>>({1, 2, 3, 4, 5});

    if (auto vec = std::any_cast<std::vector<int>>(&container)) {
        std::cout << "Vector size: " << vec->size() << "\n";
    }

    container.reset();
    std::cout << "After reset, has_value: " << container.has_value() << "\n";

    // 15. Any in Containers
    std::cout << "\n15. ANY IN CONTAINERS:\n";
    std::vector<std::any> heterogeneous;
    heterogeneous.push_back(42);
    heterogeneous.push_back(std::string("text"));
    heterogeneous.push_back(3.14);
    heterogeneous.push_back(true);

    std::cout << "Heterogeneous container contents:\n";
    for (size_t i = 0; i < heterogeneous.size(); ++i) {
        std::cout << "  Element " << i << " type: " << heterogeneous[i].type().name() << "\n";
    }

    separator("COMPARISON");

    std::cout << "\nUse OPTIONAL when:\n";
    std::cout << "- Function may or may not return a value\n";
    std::cout << "- Avoiding special 'invalid' values (like -1, nullptr)\n";
    std::cout << "- Making absence of value explicit\n";

    std::cout << "\nUse VARIANT when:\n";
    std::cout << "- Variable can be one of a fixed set of types\n";
    std::cout << "- Type-safe union needed\n";
    std::cout << "- State machines, parsers, AST nodes\n";

    std::cout << "\nUse ANY when:\n";
    std::cout << "- Need to store truly any type\n";
    std::cout << "- Type not known at compile time\n";
    std::cout << "- Interfacing with dynamic systems\n";
    std::cout << "- Be careful: runtime overhead and type checking\n";

    std::cout << "\n=== END OF OPTIONAL, VARIANT, ANY ===\n";

    return 0;
}
