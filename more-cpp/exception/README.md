# C++ Exception Handling Tutorial

Welcome! This tutorial will teach you everything about exception handling in C++ through 10 comprehensive examples.

## 📚 Learning Path

Follow these files in order for the best learning experience:

### 1. **01_basic_exception.cpp** - Fundamentals
- What are exceptions?
- Basic try-catch syntax
- How program flow changes with exceptions
- When and why to use exceptions

**Start here if**: You're completely new to exception handling

---

### 2. **02_multiple_catch.cpp** - Multiple Exception Types
- Handling different types of exceptions
- Multiple catch blocks
- Catch-all handler (...)
- Order of catch blocks

**You'll learn**: How to handle different error scenarios

---

### 3. **03_standard_exceptions.cpp** - Standard Library Exceptions
- C++ standard exception hierarchy
- Common exception types (runtime_error, logic_error, etc.)
- When to use each type
- Practical examples

**You'll learn**: All the built-in exception types C++ provides

---

### 4. **04_custom_exceptions.cpp** - Creating Your Own
- Creating custom exception classes
- Inheriting from std::exception
- Adding custom data to exceptions
- Exception hierarchies

**You'll learn**: How to create meaningful, application-specific exceptions

---

### 5. **05_rethrow_exceptions.cpp** - Exception Propagation
- Rethrowing exceptions with `throw;`
- Partial exception handling
- Exception wrapping and chaining
- Preserving exception information

**You'll learn**: How to handle exceptions at multiple levels

---

### 6. **06_nested_try_catch.cpp** - Complex Error Handling
- Nested try-catch blocks
- Exception propagation through call stack
- Stack unwinding
- Granular error handling

**You'll learn**: How exceptions propagate through your program

---

### 7. **07_constructor_exceptions.cpp** - Special Cases
- Throwing from constructors
- Why destructors should never throw
- Partially constructed objects
- Function-try-blocks

**You'll learn**: Critical rules for constructors and destructors

---

### 8. **08_exception_safety.cpp** - RAII and Safety Guarantees
- RAII (Resource Acquisition Is Initialization)
- Exception safety guarantees (no-throw, strong, basic)
- Smart pointers for automatic cleanup
- Writing exception-safe code

**You'll learn**: How to write code that doesn't leak resources

---

### 9. **09_noexcept.cpp** - Performance and Specifications
- noexcept specifier
- Conditional noexcept
- noexcept operator
- When and why to use noexcept
- Performance benefits

**You'll learn**: How to optimize code with noexcept

---

### 10. **10_real_world_example.cpp** - Putting It All Together
- Complete banking system example
- Custom exceptions in practice
- RAII with logging
- Exception-safe transactions
- Error recovery and rollback

**You'll learn**: How professionals use exceptions in real applications

---

## 🚀 How to Use This Tutorial

### Compile and Run
```bash
# Compile any file
g++ -std=c++17 -Wall -Wextra 01_basic_exception.cpp -o 01_basic_exception

# Run it
./01_basic_exception
```

### Compile all at once
```bash
# Create a build directory
mkdir -p build
cd build

# Compile all examples
g++ -std=c++17 -Wall -Wextra ../01_basic_exception.cpp -o 01_basic_exception
g++ -std=c++17 -Wall -Wextra ../02_multiple_catch.cpp -o 02_multiple_catch
g++ -std=c++17 -Wall -Wextra ../03_standard_exceptions.cpp -o 03_standard_exceptions
g++ -std=c++17 -Wall -Wextra ../04_custom_exceptions.cpp -o 04_custom_exceptions
g++ -std=c++17 -Wall -Wextra ../05_rethrow_exceptions.cpp -o 05_rethrow_exceptions
g++ -std=c++17 -Wall -Wextra ../06_nested_try_catch.cpp -o 06_nested_try_catch
g++ -std=c++17 -Wall -Wextra ../07_constructor_exceptions.cpp -o 07_constructor_exceptions
g++ -std=c++17 -Wall -Wextra ../08_exception_safety.cpp -o 08_exception_safety
g++ -std=c++17 -Wall -Wextra ../09_noexcept.cpp -o 09_noexcept
g++ -std=c++17 -Wall -Wextra ../10_real_world_example.cpp -o 10_real_world_example
```

### Using a Makefile (create this file)
```makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
BUILDDIR = build

SOURCES = $(wildcard *.cpp)
TARGETS = $(patsubst %.cpp,$(BUILDDIR)/%,$(SOURCES))

all: $(BUILDDIR) $(TARGETS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean
```

Then just run: `make` to build all, `make clean` to clean up.

---

## 📖 Study Tips

1. **Read the comments**: Each file has extensive comments explaining concepts
2. **Experiment**: Modify the code, try to break it, see what happens
3. **Uncomment dangerous lines**: Some files have commented-out code that will crash - try it!
4. **Run each example**: See the output to understand behavior
5. **Read KEY POINTS**: At the end of each file's main() function
6. **Progress sequentially**: Don't skip ahead - concepts build on each other

---

## 🎯 Important Concepts You'll Master

### Core Concepts
- ✅ try-catch-throw mechanism
- ✅ Exception types and hierarchy
- ✅ Stack unwinding
- ✅ Resource management (RAII)

### Advanced Concepts
- ✅ Exception safety guarantees
- ✅ Custom exception design
- ✅ noexcept optimization
- ✅ Exception-safe algorithms

### Best Practices
- ✅ When to use exceptions
- ✅ How to design exception hierarchies
- ✅ Writing exception-safe code
- ✅ Performance considerations

---

## 🔥 Common Pitfalls (You'll Learn to Avoid)

1. ❌ Throwing in destructors
2. ❌ Not catching by const reference
3. ❌ Using `throw e;` instead of `throw;`
4. ❌ Resource leaks with exceptions
5. ❌ Lying about noexcept
6. ❌ Catching exceptions you can't handle
7. ❌ Using exceptions for control flow

---

## 📝 After Completing This Tutorial

You should be able to:
- ✅ Write exception-safe C++ code
- ✅ Design custom exception hierarchies
- ✅ Use RAII for automatic resource management
- ✅ Understand when and how to use noexcept
- ✅ Debug exception-related issues
- ✅ Apply exception handling in real projects

---

## 🎓 Next Steps

After mastering these examples:

1. **Practice**: Write your own programs using exceptions
2. **Read**: C++ Core Guidelines on error handling
3. **Study**: Open source C++ projects to see exceptions in action
4. **Build**: Create a small project applying these concepts (like the banking example)

---

## 📚 Additional Resources

- C++ Reference: https://en.cppreference.com/w/cpp/error
- C++ Core Guidelines: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-errors
- Effective C++ (Scott Meyers) - Items on exceptions
- C++ Standard Library Exception Hierarchy

---

## 🐛 Troubleshooting

**Compilation errors?**
- Make sure you're using C++11 or later (`-std=c++17`)
- Check that you have a modern compiler (g++ 7+, clang++ 5+)

**Runtime errors?**
- That's expected! Many examples demonstrate error scenarios
- Read the output carefully to understand what's happening

---

## 💡 Pro Tips

1. Use a debugger (gdb/lldb) to step through exception handling
2. Enable all warnings (`-Wall -Wextra`) when compiling
3. Read each file's source code - comments explain everything
4. Try modifying examples to test your understanding
5. The "KEY POINTS" section at the end of each file is crucial!

---

Happy Learning! 🚀

If you have questions, review the comments in the code - they're very detailed!
