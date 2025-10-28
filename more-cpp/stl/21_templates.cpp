/**
 * 21_templates.cpp
 *
 * TEMPLATES AND GENERIC PROGRAMMING
 * - Function templates
 * - Class templates
 * - Template specialization
 * - Variadic templates
 * - SFINAE and concepts (C++20)
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

// Function template
template<typename T>
T max_value(T a, T b) {
    return (a > b) ? a : b;
}

// Multiple template parameters
template<typename T, typename U>
auto add(T a, U b) {
    return a + b;
}

// Template with default parameter
template<typename T = int>
T get_default() {
    return T{};
}

// Class template
template<typename T>
class Box {
private:
    T value;
public:
    Box(T v) : value(v) {}
    T get() const { return value; }
    void set(T v) { value = v; }
};

// Template specialization
template<>
class Box<bool> {
private:
    bool value;
public:
    Box(bool v) : value(v) {}
    bool get() const { return value; }
    void set(bool v) { value = v; }
    void toggle() { value = !value; }
};

// Variadic template
template<typename... Args>
void print_all(Args... args) {
    ((std::cout << args << " "), ...);
    std::cout << "\n";
}

template<typename T>
void print_type() {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integral type\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Floating point type\n";
    } else {
        std::cout << "Other type\n";
    }
}

// Template metaprogramming - Factorial
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// constexpr function
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// SFINAE examples
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
double_value(T value) {
    return value * 2;
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
double_value(T value) {
    return value * 2.0;
}

// Concept definition (C++20)
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T triple(T value) {
    return value * 3;
}

// Generic container operations
template<typename Container>
void print_container(const Container& cont) {
    std::cout << "Container: ";
    for (const auto& elem : cont) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

// Type deduction helper
template<typename T>
void inspect_type(T&& value) {
    std::cout << "Type is lvalue ref: " << std::is_lvalue_reference_v<T> << "\n";
    std::cout << "Type is rvalue ref: " << std::is_rvalue_reference_v<T> << "\n";
}

// Stack template class
template<typename T>
class Stack {
private:
    std::vector<T> elements;
public:
    void push(const T& elem) { elements.push_back(elem); }
    void pop() { if (!elements.empty()) elements.pop_back(); }
    T top() const { return elements.back(); }
    bool empty() const { return elements.empty(); }
    size_t size() const { return elements.size(); }
};

// Variadic sum templates
template<typename T>
T sum(T value) {
    return value;
}

template<typename T, typename... Args>
T sum(T first, Args... rest) {
    return first + sum(rest...);
}

// Count arguments template
template<typename... Args>
size_t count_args(Args... args) {
    return sizeof...(args);
}

int main() {
    std::cout << "=== TEMPLATES ===\n";

    separator("FUNCTION TEMPLATES");

    // 1. Basic Function Template
    std::cout << "\n1. FUNCTION TEMPLATE:\n";
    std::cout << "max(10, 20) = " << max_value(10, 20) << "\n";
    std::cout << "max(3.14, 2.71) = " << max_value(3.14, 2.71) << "\n";
    std::cout << "max('a', 'z') = " << max_value('a', 'z') << "\n";

    // 2. Explicit Template Arguments
    std::cout << "\n2. EXPLICIT TEMPLATE ARGUMENTS:\n";
    std::cout << "max<double>(10, 20.5) = " << max_value<double>(10, 20.5) << "\n";

    // 3. Multiple Template Parameters
    std::cout << "\n3. MULTIPLE TEMPLATE PARAMETERS:\n";
    std::cout << "add(5, 3.14) = " << add(5, 3.14) << "\n";
    std::cout << "add(std::string, char) = " << add(std::string("Hello"), '!') << "\n";

    // 4. Default Template Parameters
    std::cout << "\n4. DEFAULT TEMPLATE PARAMETERS:\n";
    std::cout << "get_default<int>() = " << get_default<int>() << "\n";
    std::cout << "get_default<double>() = " << get_default<double>() << "\n";
    std::cout << "get_default<>() = " << get_default<>() << "\n";  // Uses default

    separator("CLASS TEMPLATES");

    // 5. Basic Class Template
    std::cout << "\n5. CLASS TEMPLATE:\n";
    Box<int> int_box(42);
    Box<std::string> str_box("Hello");

    std::cout << "int_box: " << int_box.get() << "\n";
    std::cout << "str_box: " << str_box.get() << "\n";

    // 6. Template Specialization
    std::cout << "\n6. TEMPLATE SPECIALIZATION:\n";
    Box<bool> bool_box(true);
    std::cout << "bool_box: " << bool_box.get() << "\n";
    bool_box.toggle();  // Special member only for bool
    std::cout << "After toggle: " << bool_box.get() << "\n";

    // 7. Template with Container
    std::cout << "\n7. TEMPLATE WITH CONTAINER:\n";
    Stack<int> int_stack;
    int_stack.push(1);
    int_stack.push(2);
    int_stack.push(3);
    std::cout << "Stack top: " << int_stack.top() << "\n";
    std::cout << "Stack size: " << int_stack.size() << "\n";

    separator("VARIADIC TEMPLATES");

    // 8. Variadic Template
    std::cout << "\n8. VARIADIC TEMPLATE:\n";
    print_all(1, 2, 3, 4, 5);
    print_all("Hello", "World", "from", "C++");
    print_all(1, "mixed", 3.14, 'x');

    // 9. Recursive Variadic Template
    std::cout << "\n9. RECURSIVE VARIADIC:\n";
    std::cout << "sum(1, 2, 3, 4, 5) = " << sum(1, 2, 3, 4, 5) << "\n";
    std::cout << "sum(1.5, 2.5, 3.5) = " << sum(1.5, 2.5, 3.5) << "\n";

    // 10. sizeof... Operator
    std::cout << "\n10. SIZEOF... OPERATOR:\n";
    std::cout << "count_args(1, 2, 3) = " << count_args(1, 2, 3) << "\n";
    std::cout << "count_args('a', 'b', 'c', 'd', 'e') = " << count_args('a', 'b', 'c', 'd', 'e') << "\n";

    separator("CONSTEXPR IF (C++17)");

    // 11. constexpr if
    std::cout << "\n11. CONSTEXPR IF:\n";
    print_type<int>();
    print_type<double>();
    print_type<std::string>();

    separator("TEMPLATE METAPROGRAMMING");

    // 12. Compile-time Factorial
    std::cout << "\n12. COMPILE-TIME FACTORIAL:\n";
    std::cout << "Factorial<5> = " << Factorial<5>::value << "\n";
    std::cout << "Factorial<10> = " << Factorial<10>::value << "\n";

    // 13. constexpr Function (C++11)
    std::cout << "\n13. CONSTEXPR FUNCTION:\n";
    constexpr int fact5 = factorial(5);  // Computed at compile time
    std::cout << "factorial(5) = " << fact5 << "\n";
    std::cout << "factorial(7) = " << factorial(7) << "\n";

    separator("SFINAE");

    // 14. SFINAE (Substitution Failure Is Not An Error)
    std::cout << "\n14. SFINAE:\n";
    std::cout << "double_value(5) = " << double_value(5) << "\n";
    std::cout << "double_value(3.14) = " << double_value(3.14) << "\n";

    separator("CONCEPTS (C++20)");

    std::cout << "\n15. CONCEPTS (C++20):\n";
    // Requires C++20 compiler support
    std::cout << "triple(10) = " << triple(10) << "\n";
    std::cout << "triple(3.14) = " << triple(3.14) << "\n";
    // triple("hello");  // Compile error: doesn't satisfy Numeric

    separator("PRACTICAL EXAMPLES");

    // 16. Generic Container Operations
    std::cout << "\n16. GENERIC CONTAINER OPERATIONS:\n";
    std::vector<int> vec = {1, 2, 3, 4, 5};
    print_container(vec);

    // 17. Type Deduction
    std::cout << "\n17. TYPE DEDUCTION:\n";
    int x = 42;
    inspect_type(x);          // lvalue
    inspect_type(42);         // rvalue

    separator("BEST PRACTICES");

    std::cout << "\n1. Use templates for type-safe generic code\n";
    std::cout << "2. Prefer concepts (C++20) over SFINAE\n";
    std::cout << "3. Use constexpr for compile-time computation\n";
    std::cout << "4. Document template requirements\n";
    std::cout << "5. Use auto and decltype for type deduction\n";
    std::cout << "6. Consider compilation time impact\n";
    std::cout << "7. Provide clear error messages (concepts help)\n";
    std::cout << "8. Use template specialization sparingly\n";

    std::cout << "\n=== END OF TEMPLATES ===\n";

    return 0;
}
