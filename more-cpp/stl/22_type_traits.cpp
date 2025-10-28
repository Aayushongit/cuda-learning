/**
 * 22_type_traits.cpp
 *
 * TYPE TRAITS AND METAPROGRAMMING
 * - Type properties
 * - Type transformations
 * - Type relationships
 * - Compile-time type checking
 * - SFINAE applications
 */

#include <iostream>
#include <type_traits>
#include <string>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

// SFINAE print_type overloads
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
print_type(T value) {
    std::cout << "Integer: " << value << "\n";
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
print_type(T value) {
    std::cout << "Float: " << value << "\n";
}

// Conditional type alias
template<typename T>
using LargeType = typename std::conditional<sizeof(T) >= 8, T, long>::type;

// Generic safe_divide template
template<typename T>
T safe_divide(T a, T b) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");

    if constexpr (std::is_integral_v<T>) {
        return (b != 0) ? a / b : 0;
    } else {
        return (b != 0.0) ? a / b : 0.0;
    }
}

int main() {
    std::cout << "=== TYPE TRAITS ===\n";

    separator("PRIMARY TYPE CATEGORIES");

    // 1. is_integral, is_floating_point
    std::cout << "\n1. BASIC TYPE CHECKS:\n";
    std::cout << "is_integral<int>: " << std::is_integral<int>::value << "\n";
    std::cout << "is_integral<double>: " << std::is_integral<double>::value << "\n";
    std::cout << "is_floating_point<float>: " << std::is_floating_point<float>::value << "\n";
    std::cout << "is_floating_point<int>: " << std::is_floating_point<int>::value << "\n";

    // 2. is_arithmetic, is_fundamental
    std::cout << "\n2. COMPOSITE TYPE CHECKS:\n";
    std::cout << "is_arithmetic<int>: " << std::is_arithmetic<int>::value << "\n";
    std::cout << "is_arithmetic<float>: " << std::is_arithmetic<float>::value << "\n";
    std::cout << "is_fundamental<int>: " << std::is_fundamental<int>::value << "\n";
    std::cout << "is_fundamental<std::string>: " << std::is_fundamental<std::string>::value << "\n";

    // 3. is_array, is_pointer
    std::cout << "\n3. COMPOUND TYPE CHECKS:\n";
    std::cout << "is_array<int[]>: " << std::is_array<int[]>::value << "\n";
    std::cout << "is_array<int[5]>: " << std::is_array<int[5]>::value << "\n";
    std::cout << "is_array<int>: " << std::is_array<int>::value << "\n";
    std::cout << "is_pointer<int*>: " << std::is_pointer<int*>::value << "\n";
    std::cout << "is_pointer<int>: " << std::is_pointer<int>::value << "\n";

    // 4. is_class, is_enum
    std::cout << "\n4. CLASS AND ENUM CHECKS:\n";
    struct MyStruct {};
    enum MyEnum { A, B, C };

    std::cout << "is_class<MyStruct>: " << std::is_class<MyStruct>::value << "\n";
    std::cout << "is_class<int>: " << std::is_class<int>::value << "\n";
    std::cout << "is_enum<MyEnum>: " << std::is_enum<MyEnum>::value << "\n";
    std::cout << "is_enum<int>: " << std::is_enum<int>::value << "\n";

    separator("TYPE PROPERTIES");

    // 5. const and volatile
    std::cout << "\n5. CONST/VOLATILE CHECKS:\n";
    std::cout << "is_const<const int>: " << std::is_const<const int>::value << "\n";
    std::cout << "is_const<int>: " << std::is_const<int>::value << "\n";
    std::cout << "is_volatile<volatile int>: " << std::is_volatile<volatile int>::value << "\n";

    // 6. References
    std::cout << "\n6. REFERENCE CHECKS:\n";
    std::cout << "is_lvalue_reference<int&>: " << std::is_lvalue_reference<int&>::value << "\n";
    std::cout << "is_rvalue_reference<int&&>: " << std::is_rvalue_reference<int&&>::value << "\n";
    std::cout << "is_reference<int&>: " << std::is_reference<int&>::value << "\n";
    std::cout << "is_reference<int>: " << std::is_reference<int>::value << "\n";

    // 7. Signed/Unsigned
    std::cout << "\n7. SIGN CHECKS:\n";
    std::cout << "is_signed<int>: " << std::is_signed<int>::value << "\n";
    std::cout << "is_unsigned<unsigned int>: " << std::is_unsigned<unsigned int>::value << "\n";
    std::cout << "is_signed<float>: " << std::is_signed<float>::value << "\n";

    separator("TYPE RELATIONSHIPS");

    // 8. is_same
    std::cout << "\n8. IS_SAME:\n";
    std::cout << "is_same<int, int>: " << std::is_same<int, int>::value << "\n";
    std::cout << "is_same<int, float>: " << std::is_same<int, float>::value << "\n";
    std::cout << "is_same<int, const int>: " << std::is_same<int, const int>::value << "\n";

    // 9. is_base_of, is_convertible
    std::cout << "\n9. IS_BASE_OF, IS_CONVERTIBLE:\n";
    class Base {};
    class Derived : public Base {};

    std::cout << "is_base_of<Base, Derived>: " << std::is_base_of<Base, Derived>::value << "\n";
    std::cout << "is_base_of<Derived, Base>: " << std::is_base_of<Derived, Base>::value << "\n";
    std::cout << "is_convertible<Derived*, Base*>: " << std::is_convertible<Derived*, Base*>::value << "\n";
    std::cout << "is_convertible<int, double>: " << std::is_convertible<int, double>::value << "\n";

    separator("TYPE TRANSFORMATIONS");

    // 10. remove_const, remove_reference
    std::cout << "\n10. REMOVE QUALIFIERS:\n";
    using ConstInt = const int;
    using IntRef = int&;

    std::cout << "is_same<remove_const<ConstInt>::type, int>: "
              << std::is_same<std::remove_const<ConstInt>::type, int>::value << "\n";

    std::cout << "is_same<remove_reference<IntRef>::type, int>: "
              << std::is_same<std::remove_reference<IntRef>::type, int>::value << "\n";

    // 11. remove_cv, remove_cvref (C++20)
    std::cout << "\n11. REMOVE_CV, REMOVE_CVREF:\n";
    using CVInt = const volatile int;
    std::cout << "is_same<remove_cv<CVInt>::type, int>: "
              << std::is_same<std::remove_cv<CVInt>::type, int>::value << "\n";

    using CVRefInt = const volatile int&;
    std::cout << "is_same<remove_cvref<CVRefInt>::type, int>: "
              << std::is_same<std::remove_cvref<CVRefInt>::type, int>::value << "\n";

    // 12. add_const, add_pointer
    std::cout << "\n12. ADD QUALIFIERS:\n";
    std::cout << "is_same<add_const<int>::type, const int>: "
              << std::is_same<std::add_const<int>::type, const int>::value << "\n";

    std::cout << "is_same<add_pointer<int>::type, int*>: "
              << std::is_same<std::add_pointer<int>::type, int*>::value << "\n";

    // 13. decay
    std::cout << "\n13. DECAY:\n";
    std::cout << "is_same<decay<int&>::type, int>: "
              << std::is_same<std::decay<int&>::type, int>::value << "\n";

    std::cout << "is_same<decay<const int&>::type, int>: "
              << std::is_same<std::decay<const int&>::type, int>::value << "\n";

    std::cout << "is_same<decay<int[10]>::type, int*>: "
              << std::is_same<std::decay<int[10]>::type, int*>::value << "\n";

    separator("CONSTRUCTIBILITY");

    // 14. is_constructible, is_default_constructible
    std::cout << "\n14. CONSTRUCTIBILITY:\n";
    struct NonDefault {
        NonDefault(int) {}
    };

    std::cout << "is_default_constructible<int>: "
              << std::is_default_constructible<int>::value << "\n";

    std::cout << "is_default_constructible<NonDefault>: "
              << std::is_default_constructible<NonDefault>::value << "\n";

    std::cout << "is_constructible<NonDefault, int>: "
              << std::is_constructible<NonDefault, int>::value << "\n";

    // 15. is_copy_constructible, is_move_constructible
    std::cout << "\n15. COPY/MOVE CONSTRUCTIBILITY:\n";
    struct NoCopy {
        NoCopy(const NoCopy&) = delete;
    };

    std::cout << "is_copy_constructible<int>: "
              << std::is_copy_constructible<int>::value << "\n";

    std::cout << "is_copy_constructible<NoCopy>: "
              << std::is_copy_constructible<NoCopy>::value << "\n";

    std::cout << "is_move_constructible<int>: "
              << std::is_move_constructible<int>::value << "\n";

    separator("PRACTICAL SFINAE EXAMPLES");

    // 16. enable_if
    std::cout << "\n16. ENABLE_IF:\n";
    print_type(42);
    print_type(3.14);

    // 17. conditional
    std::cout << "\n17. CONDITIONAL:\n";
    std::cout << "LargeType<char> is long: "
              << std::is_same<LargeType<char>, long>::value << "\n";

    std::cout << "LargeType<long long> is long long: "
              << std::is_same<LargeType<long long>, long long>::value << "\n";

    separator("COMPILE-TIME CONDITIONALS");

    // 18. constexpr if with type traits
    std::cout << "\n18. CONSTEXPR IF WITH TYPE TRAITS:\n";
    auto process = [](auto value) {
        using T = decltype(value);
        if constexpr (std::is_integral_v<T>) {
            std::cout << "Processing integer: " << value * 2 << "\n";
        } else if constexpr (std::is_floating_point_v<T>) {
            std::cout << "Processing float: " << value * 2.0 << "\n";
        } else {
            std::cout << "Processing other type\n";
        }
    };

    process(10);
    process(3.14);
    process("hello");

    separator("SIZE AND ALIGNMENT");

    // 19. sizeof, alignof
    std::cout << "\n19. SIZE AND ALIGNMENT:\n";
    std::cout << "sizeof(int): " << sizeof(int) << "\n";
    std::cout << "sizeof(double): " << sizeof(double) << "\n";
    std::cout << "alignof(int): " << alignof(int) << "\n";
    std::cout << "alignof(double): " << alignof(double) << "\n";

    struct Aligned {
        alignas(16) int value;
    };
    std::cout << "alignof(Aligned): " << alignof(Aligned) << "\n";

    separator("PRACTICAL APPLICATIONS");

    // 20. Generic Algorithm with Type Checks
    std::cout << "\n20. GENERIC ALGORITHM:\n";
    std::cout << "safe_divide(10, 3) = " << safe_divide(10, 3) << "\n";
    std::cout << "safe_divide(10.0, 3.0) = " << safe_divide(10.0, 3.0) << "\n";

    separator("BEST PRACTICES");

    std::cout << "\n1. Use _v suffix for value (C++17): is_integral_v<T>\n";
    std::cout << "2. Use _t suffix for type (C++14): remove_const_t<T>\n";
    std::cout << "3. Prefer constexpr if over SFINAE when possible\n";
    std::cout << "4. Use static_assert for compile-time checks\n";
    std::cout << "5. Document template requirements with concepts (C++20)\n";
    std::cout << "6. Type traits enable generic, type-safe code\n";
    std::cout << "7. Combine with templates for powerful metaprogramming\n";

    std::cout << "\n=== END OF TYPE TRAITS ===\n";

    return 0;
}
