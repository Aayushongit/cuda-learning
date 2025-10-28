# C++ Types, Templates, and Structs Comprehensive Guide

## Building and Running

```bash
# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run individual demos
./typedef_demo
./using_demo
./struct_demo
./function_templates_demo
./class_templates_demo
./specialization_demo
./combined_demo
```

## Files Overview

1. **01_typedef_examples.cpp** - Old C-style type aliasing
2. **02_using_examples.cpp** - Modern C++11 type aliases (preferred)
3. **03_struct_examples.cpp** - Struct usage, PODs, inheritance
4. **04_function_templates.cpp** - Generic functions, variadic templates
5. **05_class_templates.cpp** - Generic classes, containers
6. **06_template_specialization.cpp** - Customizing behavior for specific types
7. **07_combining_all.cpp** - Real-world patterns combining all concepts

## Key Takeaways

- **Prefer `using` over `typedef`** - More readable and powerful
- **Structs vs Classes** - Only difference is default access (public vs private)
- **Templates are compile-time** - Zero runtime overhead
- **Specialization** - Customize generic code for specific types
