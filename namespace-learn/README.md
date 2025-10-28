# C++ Namespace Learning Examples

This directory contains 10 comprehensive examples to learn C++ namespaces.

## Examples Overview

1. **01_basic_namespace.cpp** - Basic namespace creation and usage
2. **02_nested_namespace.cpp** - Nested namespaces (traditional and C++17 syntax)
3. **03_multiple_namespaces.cpp** - Multiple namespaces with same function names
4. **04_namespace_alias.cpp** - Creating aliases for long namespace names
5. **05_using_directive.cpp** - Using directive and using declaration
6. **06_anonymous_namespace.cpp** - Anonymous/unnamed namespaces for file-private code
7. **07_main.cpp + 07_geometry.cpp/h** - Namespaces across multiple files
8. **08_namespace_classes.cpp** - Classes inside namespaces
9. **09_inline_namespace.cpp** - Inline namespaces for API versioning
10. **10_real_world_example.cpp** - Real-world project structure

## How to Compile and Run

### Compile individual examples:
```bash
g++ 01_basic_namespace.cpp -o 01_program && ./01_program
g++ 02_nested_namespace.cpp -o 02_program && ./02_program
g++ 03_multiple_namespaces.cpp -o 03_program && ./03_program
g++ 04_namespace_alias.cpp -o 04_program && ./04_program
g++ 05_using_directive.cpp -o 05_program && ./05_program
g++ 06_anonymous_namespace.cpp -o 06_program && ./06_program
g++ 08_namespace_classes.cpp -o 08_program && ./08_program
g++ 09_inline_namespace.cpp -o 09_program && ./09_program
g++ 10_real_world_example.cpp -o 10_program && ./10_program
```

### Example 7 (multiple files):
```bash
g++ 07_main.cpp 07_geometry.cpp -o 07_program && ./07_program
```

### Or use the compile script:
```bash
chmod +x compile_and_run.sh
./compile_and_run.sh
```

## Learning Path

Start with example 01 and work your way through to example 10. Each builds on previous concepts.
