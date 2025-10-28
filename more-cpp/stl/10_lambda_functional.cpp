/**
 * 10_lambda_functional.cpp
 *
 * LAMBDA EXPRESSIONS AND FUNCTIONAL PROGRAMMING
 * - Lambda syntax and captures
 * - std::function
 * - std::bind
 * - Function objects (functors)
 * - Functional programming patterns
 */

#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== LAMBDA EXPRESSIONS AND FUNCTIONAL PROGRAMMING ===\n";

    // ========== LAMBDA BASICS ==========
    separator("LAMBDA BASICS");

    // 1. Basic Lambda
    std::cout << "\n1. BASIC LAMBDA:\n";
    auto simple = []() {
        std::cout << "Hello from lambda!\n";
    };
    simple();

    // Lambda with parameters
    auto add = [](int a, int b) {
        return a + b;
    };
    std::cout << "add(3, 4) = " << add(3, 4) << "\n";

    // Explicit return type
    auto divide = [](double a, double b) -> double {
        return b != 0 ? a / b : 0;
    };
    std::cout << "divide(10, 3) = " << divide(10, 3) << "\n";

    // 2. Lambda Captures
    std::cout << "\n2. LAMBDA CAPTURES:\n";

    int x = 10, y = 20;

    // Capture by value
    auto capture_value = [x, y]() {
        std::cout << "Captured by value: x=" << x << ", y=" << y << "\n";
        // x = 100;  // ERROR: Cannot modify captured value
    };
    capture_value();

    // Capture by reference
    auto capture_ref = [&x, &y]() {
        std::cout << "Captured by reference: x=" << x << ", y=" << y << "\n";
        x = 100;  // OK: Can modify
    };
    capture_ref();
    std::cout << "After capture_ref, x=" << x << "\n";

    // Capture all by value
    auto capture_all_val = [=]() {
        std::cout << "Capture all by value: x=" << x << ", y=" << y << "\n";
    };
    capture_all_val();

    // Capture all by reference
    auto capture_all_ref = [&]() {
        x = 200;
        y = 300;
    };
    capture_all_ref();
    std::cout << "After capture_all_ref: x=" << x << ", y=" << y << "\n";

    // Mixed captures
    int a = 1, b = 2, c = 3;
    auto mixed = [a, &b, &c]() {  // a by value, b and c by ref
        std::cout << "a=" << a << ", b=" << b << ", c=" << c << "\n";
    };
    mixed();

    // 3. Mutable Lambdas
    std::cout << "\n3. MUTABLE LAMBDAS:\n";
    int counter = 0;
    auto increment = [counter]() mutable {
        counter++;
        std::cout << "Inside lambda, counter=" << counter << "\n";
        return counter;
    };

    increment();
    increment();
    std::cout << "Outside lambda, counter=" << counter << "\n";  // Still 0

    // 4. Init Captures (C++14)
    std::cout << "\n4. INIT CAPTURES:\n";
    auto init_cap = [value = 42, square = 10 * 10]() {
        std::cout << "Init captures: value=" << value << ", square=" << square << "\n";
    };
    init_cap();

    // Move capture
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto move_cap = [v = std::move(vec)]() {
        std::cout << "Moved vector size: " << v.size() << "\n";
    };
    move_cap();
    std::cout << "Original vector size after move: " << vec.size() << "\n";

    // 5. Generic Lambdas (C++14)
    std::cout << "\n5. GENERIC LAMBDAS:\n";
    auto generic = [](auto a, auto b) {
        return a + b;
    };
    std::cout << "generic(5, 10) = " << generic(5, 10) << "\n";
    std::cout << "generic(3.5, 2.5) = " << generic(3.5, 2.5) << "\n";
    std::cout << "generic(string) = " << generic(std::string("Hello "), std::string("World")) << "\n";

    // 6. Lambdas with Algorithms
    std::cout << "\n6. LAMBDAS WITH ALGORITHMS:\n";
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // for_each
    std::cout << "Squares: ";
    std::for_each(numbers.begin(), numbers.end(), [](int n) {
        std::cout << n * n << " ";
    });
    std::cout << "\n";

    // count_if
    int even_count = std::count_if(numbers.begin(), numbers.end(),
                                    [](int n) { return n % 2 == 0; });
    std::cout << "Even count: " << even_count << "\n";

    // find_if
    auto it = std::find_if(numbers.begin(), numbers.end(),
                           [](int n) { return n > 7; });
    std::cout << "First element > 7: " << *it << "\n";

    // transform
    std::vector<int> doubled(numbers.size());
    std::transform(numbers.begin(), numbers.end(), doubled.begin(),
                   [](int n) { return n * 2; });
    std::cout << "Doubled: ";
    for (int n : doubled) std::cout << n << " ";
    std::cout << "\n";

    // sort with custom comparator
    std::vector<int> to_sort = {5, 2, 8, 1, 9};
    std::sort(to_sort.begin(), to_sort.end(), [](int a, int b) {
        return a > b;  // Descending
    });
    std::cout << "Sorted descending: ";
    for (int n : to_sort) std::cout << n << " ";
    std::cout << "\n";

    // ========== STD::FUNCTION ==========
    separator("STD::FUNCTION");

    // 7. std::function Basics
    std::cout << "\n7. STD::FUNCTION BASICS:\n";

    std::function<int(int, int)> func_add = [](int a, int b) { return a + b; };
    std::cout << "func_add(10, 20) = " << func_add(10, 20) << "\n";

    // Can reassign
    func_add = [](int a, int b) { return a * b; };
    std::cout << "func_add(10, 20) = " << func_add(10, 20) << " (multiply)\n";

    // 8. Storing Different Callables
    std::cout << "\n8. STORING DIFFERENT CALLABLES:\n";

    // Regular function
    auto regular_func = [](int x) { return x * x; };

    // Function object
    struct Multiplier {
        int operator()(int x) const { return x * 3; }
    };

    std::function<int(int)> callable;

    callable = regular_func;
    std::cout << "Lambda: " << callable(5) << "\n";

    callable = Multiplier();
    std::cout << "Functor: " << callable(5) << "\n";

    // 9. Function as Parameter
    std::cout << "\n9. FUNCTION AS PARAMETER:\n";

    auto apply = [](std::function<int(int)> f, int value) {
        return f(value);
    };

    std::cout << "apply(square, 7) = " << apply([](int x) { return x * x; }, 7) << "\n";
    std::cout << "apply(double, 7) = " << apply([](int x) { return x * 2; }, 7) << "\n";

    // 10. Storing Functions
    std::cout << "\n10. STORING FUNCTIONS:\n";

    std::vector<std::function<int(int)>> operations;
    operations.push_back([](int x) { return x + 1; });
    operations.push_back([](int x) { return x * 2; });
    operations.push_back([](int x) { return x * x; });

    int value = 5;
    std::cout << "Applying operations to " << value << ":\n";
    for (const auto& op : operations) {
        std::cout << op(value) << " ";
    }
    std::cout << "\n";

    // ========== STD::BIND ==========
    separator("STD::BIND");

    // 11. std::bind Basics
    std::cout << "\n11. STD::BIND BASICS:\n";

    auto multiply = [](int a, int b) { return a * b; };

    // Bind first argument
    auto times_two = std::bind(multiply, 2, std::placeholders::_1);
    std::cout << "times_two(5) = " << times_two(5) << "\n";

    // Bind second argument
    auto times_three = std::bind(multiply, std::placeholders::_1, 3);
    std::cout << "times_three(5) = " << times_three(5) << "\n";

    // 12. Reordering Arguments
    std::cout << "\n12. REORDERING ARGUMENTS:\n";

    auto subtract = [](int a, int b) { return a - b; };

    auto reversed_subtract = std::bind(subtract,
                                       std::placeholders::_2,
                                       std::placeholders::_1);
    std::cout << "subtract(10, 3) = " << subtract(10, 3) << "\n";
    std::cout << "reversed_subtract(10, 3) = " << reversed_subtract(10, 3) << "\n";

    // ========== FUNCTORS ==========
    separator("FUNCTORS (FUNCTION OBJECTS)");

    // 13. Custom Functor
    std::cout << "\n13. CUSTOM FUNCTOR:\n";

    struct Accumulator {
        int sum = 0;
        void operator()(int value) {
            sum += value;
        }
    };

    Accumulator acc;
    std::vector<int> vals = {1, 2, 3, 4, 5};
    acc = std::for_each(vals.begin(), vals.end(), acc);
    std::cout << "Accumulated sum: " << acc.sum << "\n";

    // 14. Stateful Functor
    std::cout << "\n14. STATEFUL FUNCTOR:\n";

    struct Counter {
        int count = 0;
        bool operator()(int) {
            return ++count <= 3;
        }
    };

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    auto end_it = std::find_if_not(data.begin(), data.end(), Counter());
    std::cout << "First 3 elements: ";
    for (auto it = data.begin(); it != end_it; ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    // ========== FUNCTIONAL PROGRAMMING PATTERNS ==========
    separator("FUNCTIONAL PROGRAMMING PATTERNS");

    // 15. Map-Reduce
    std::cout << "\n15. MAP-REDUCE:\n";
    std::vector<int> input = {1, 2, 3, 4, 5};

    // Map: square each element
    std::vector<int> mapped(input.size());
    std::transform(input.begin(), input.end(), mapped.begin(),
                   [](int x) { return x * x; });

    // Reduce: sum all elements
    int sum = std::accumulate(mapped.begin(), mapped.end(), 0);
    std::cout << "Sum of squares: " << sum << "\n";

    // 16. Filter-Map-Reduce
    std::cout << "\n16. FILTER-MAP-REDUCE:\n";
    std::vector<int> source = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Filter evens
    std::vector<int> filtered;
    std::copy_if(source.begin(), source.end(), std::back_inserter(filtered),
                 [](int x) { return x % 2 == 0; });

    // Map to squares
    std::transform(filtered.begin(), filtered.end(), filtered.begin(),
                   [](int x) { return x * x; });

    // Reduce to sum
    int result = std::accumulate(filtered.begin(), filtered.end(), 0);
    std::cout << "Sum of squares of evens: " << result << "\n";

    // 17. Compose Functions
    std::cout << "\n17. COMPOSE FUNCTIONS:\n";

    auto compose = [](auto f, auto g) {
        return [=](auto x) { return f(g(x)); };
    };

    auto add_one = [](int x) { return x + 1; };
    auto square = [](int x) { return x * x; };

    auto square_then_add = compose(add_one, square);
    auto add_then_square = compose(square, add_one);

    std::cout << "square_then_add(3) = " << square_then_add(3) << "\n";  // 3² + 1 = 10
    std::cout << "add_then_square(3) = " << add_then_square(3) << "\n";  // (3+1)² = 16

    // 18. Currying
    std::cout << "\n18. CURRYING:\n";

    auto curry_add = [](int a) {
        return [a](int b) {
            return a + b;
        };
    };

    auto add_5 = curry_add(5);
    std::cout << "add_5(10) = " << add_5(10) << "\n";
    std::cout << "add_5(20) = " << add_5(20) << "\n";

    // 19. Higher-Order Functions
    std::cout << "\n19. HIGHER-ORDER FUNCTIONS:\n";

    auto apply_twice = [](auto f, int x) {
        return f(f(x));
    };

    auto increment_by_one = [](int x) { return x + 1; };
    std::cout << "apply_twice(increment_by_one, 5) = " << apply_twice(increment_by_one, 5) << "\n";

    // ========== PRACTICAL EXAMPLES ==========
    separator("PRACTICAL EXAMPLES");

    // 20. Event Handler System
    std::cout << "\n20. EVENT HANDLER SYSTEM:\n";

    std::vector<std::function<void(int)>> event_handlers;

    event_handlers.push_back([](int event) {
        std::cout << "Handler 1 received event: " << event << "\n";
    });

    event_handlers.push_back([](int event) {
        std::cout << "Handler 2 received event: " << event << "\n";
    });

    // Trigger event
    std::cout << "Triggering event 42:\n";
    for (const auto& handler : event_handlers) {
        handler(42);
    }

    std::cout << "\n=== END OF LAMBDA AND FUNCTIONAL PROGRAMMING ===\n";

    return 0;
}
