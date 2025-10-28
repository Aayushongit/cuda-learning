# C++ STL and Standard Library - Zero to Hero

A comprehensive collection of 24 examples covering all major C++ Standard Library features, from beginner to advanced topics.

## Building the Examples

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build all examples
cmake --build .

# Or build specific example
cmake --build . --target 01_vector_basics

# Run an example
./01_vector_basics
```

## Topics Covered

### Containers (01-06)

- **01_vector_basics.cpp** - Dynamic arrays, most commonly used container
- **02_array_deque.cpp** - Fixed-size arrays and double-ended queues
- **03_list_forward_list.cpp** - Doubly and singly-linked lists
- **04_set_map.cpp** - Ordered associative containers (trees)
- **05_unordered_containers.cpp** - Hash tables for O(1) lookups
- **06_container_adapters.cpp** - Stack, queue, priority_queue

### Algorithms and Iteration (07-09)

- **07_iterators.cpp** - Iterator types, operations, and patterns
- **08_algorithms_part1.cpp** - Sorting, searching, modifying sequences
- **09_algorithms_part2.cpp** - Numeric algorithms, heaps, permutations

### Modern C++ Features (10-12)

- **10_lambda_functional.cpp** - Lambdas, std::function, functional programming
- **11_smart_pointers.cpp** - unique_ptr, shared_ptr, weak_ptr
- **12_strings.cpp** - String manipulation and string_view

### I/O and Text Processing (13-14)

- **13_streams_io.cpp** - File I/O, console I/O, string streams
- **14_regex.cpp** - Regular expressions for pattern matching

### Time and Concurrency (15-16)

- **15_chrono_time.cpp** - Time measurement, durations, clocks
- **16_threading.cpp** - Threads, mutexes, futures, atomics

### Advanced Features (17-20)

- **17_filesystem.cpp** - File and directory operations (C++17)
- **18_optional_variant_any.cpp** - Modern type-safe alternatives
- **19_tuple_pair.cpp** - Heterogeneous collections
- **20_exception_handling.cpp** - Exception safety and best practices

### Expert Topics (21-24)

- **21_templates.cpp** - Generic programming, variadic templates
- **22_type_traits.cpp** - Compile-time type introspection
- **23_move_semantics.cpp** - Perfect forwarding, Rule of Five
- **24_ranges_views.cpp** - Modern range algorithms (C++20)

## Learning Path

### Beginner (Start Here!)
1. Vector Basics (01)
2. Algorithms Part 1 (08)
3. Strings (12)
4. Streams & I/O (13)
5. Exception Handling (20)

### Intermediate
6. Array & Deque (02)
7. Set & Map (04)
8. Unordered Containers (05)
9. Smart Pointers (11)
10. Lambda & Functional (10)

### Advanced
11. Iterators (07)
12. Algorithms Part 2 (09)
13. Regex (14)
14. Chrono & Time (15)
15. Optional, Variant, Any (18)

### Expert
16. Threading (16)
17. Filesystem (17)
18. Tuple & Pair (19)
19. Templates (21)
20. Type Traits (22)
21. Move Semantics (23)
22. Ranges & Views (24)

## Compiler Requirements

- **Minimum**: C++17 (most features)
- **Recommended**: C++20 (full support for ranges, concepts)
- **Tested with**: GCC 9+, Clang 10+, MSVC 2019+

## Quick Reference

### Most Used Containers
- `std::vector` - Dynamic array (default choice)
- `std::unordered_map` - Fast key-value lookups
- `std::string` - Text storage and manipulation
- `std::set` - Unique sorted elements

### Most Used Algorithms
- `std::sort` - Sort elements
- `std::find` / `std::find_if` - Search
- `std::transform` - Apply function to range
- `std::copy` / `std::copy_if` - Copy elements

### Modern C++ Must-Know
- Smart pointers (`unique_ptr`, `shared_ptr`)
- Lambda expressions
- Move semantics (`std::move`)
- `auto` keyword for type deduction
- Range-based for loops
- Structured bindings (C++17)

## Tips for Learning

1. **Run the examples** - Compile and execute to see output
2. **Modify the code** - Change values, add your own experiments
3. **Read the comments** - Each file has extensive documentation
4. **Practice** - Implement your own versions of concepts
5. **Reference** - Use these as quick reference when coding

## Additional Resources

- [cppreference.com](https://en.cppreference.com/) - Comprehensive reference
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/) - Best practices
- [Compiler Explorer](https://godbolt.org/) - See generated assembly

## Common Patterns

### Container Selection
```cpp
// Default choice
std::vector<T>

// Need fast lookup by key
std::unordered_map<Key, Value>

// Need sorted unique elements
std::set<T>

// Need stack/queue behavior
std::stack<T> / std::queue<T>
```

### Algorithm Usage
```cpp
// Sort
std::sort(vec.begin(), vec.end());

// Find
auto it = std::find(vec.begin(), vec.end(), value);

// Transform
std::transform(src.begin(), src.end(), dest.begin(), func);

// Filter (erase-remove idiom)
vec.erase(std::remove_if(vec.begin(), vec.end(), pred), vec.end());
```

### Modern C++ Patterns
```cpp
// Smart pointers
auto ptr = std::make_unique<T>(args...);
auto shared = std::make_shared<T>(args...);

// Optional return
std::optional<T> find_value(...);

// Variant (type-safe union)
std::variant<int, double, std::string> value;

// Structured bindings
auto [key, value] = map.insert({k, v});
```

## Next Steps

After mastering these topics:
1. Study design patterns
2. Learn about performance optimization
3. Explore C++23 features
4. Practice with real projects
5. Read production codebases

Happy coding! 🚀
