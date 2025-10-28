# Quick Start Guide - C++ Exception Handling Tutorial

## 🚀 Get Started in 3 Steps

### Step 1: Compile All Examples
```bash
cd "/media/dell/Hard Drive/Summer of code/Robotics/ROS2/ros2-learn/more-cpp/exception/"

# Compile all at once
g++ -std=c++17 -Wall -Wextra 01_basic_exception.cpp -o build/01_basic_exception
g++ -std=c++17 -Wall -Wextra 02_multiple_catch.cpp -o build/02_multiple_catch
g++ -std=c++17 -Wall -Wextra 03_standard_exceptions.cpp -o build/03_standard_exceptions
g++ -std=c++17 -Wall -Wextra 04_custom_exceptions.cpp -o build/04_custom_exceptions
g++ -std=c++17 -Wall -Wextra 05_rethrow_exceptions.cpp -o build/05_rethrow_exceptions
g++ -std=c++17 -Wall -Wextra 06_nested_try_catch.cpp -o build/06_nested_try_catch
g++ -std=c++17 -Wall -Wextra 07_constructor_exceptions.cpp -o build/07_constructor_exceptions
g++ -std=c++17 -Wall -Wextra 08_exception_safety.cpp -o build/08_exception_safety
g++ -std=c++17 -Wall -Wextra 09_noexcept.cpp -o build/09_noexcept
g++ -std=c++17 -Wall -Wextra 10_real_world_example.cpp -o build/10_real_world_example
```

### Step 2: Run Examples
```bash
# Run them in order
./build/01_basic_exception
./build/02_multiple_catch
./build/03_standard_exceptions
./build/04_custom_exceptions
./build/05_rethrow_exceptions
./build/06_nested_try_catch
./build/07_constructor_exceptions
./build/08_exception_safety
./build/09_noexcept
./build/10_real_world_example
```

### Step 3: Learn!
Open each `.cpp` file and read the extensive comments while running the examples.

---

## 📖 What Each File Teaches You

| File | Topic | Time |
|------|-------|------|
| `01_basic_exception.cpp` | Fundamentals of try-catch-throw | 10 min |
| `02_multiple_catch.cpp` | Handling multiple exception types | 10 min |
| `03_standard_exceptions.cpp` | Standard library exceptions | 15 min |
| `04_custom_exceptions.cpp` | Creating your own exceptions | 15 min |
| `05_rethrow_exceptions.cpp` | Rethrowing and exception wrapping | 15 min |
| `06_nested_try_catch.cpp` | Nested exception handling | 15 min |
| `07_constructor_exceptions.cpp` | Exceptions in constructors | 20 min |
| `08_exception_safety.cpp` | RAII and exception safety | 20 min |
| `09_noexcept.cpp` | noexcept specifier | 15 min |
| `10_real_world_example.cpp` | Complete banking system | 30 min |

**Total learning time:** ~2.5 hours

---

## 💡 Tips for Learning

1. **Read the code first**, then run it
2. **Modify the examples** to test your understanding
3. **Uncomment dangerous code** (marked in comments) to see what happens
4. **Read the "KEY POINTS"** section at the end of each main() function
5. **Try the exercises** suggested in comments

---

## 🎯 Learning Objectives

After completing this tutorial, you'll be able to:
- ✅ Write exception-safe C++ code
- ✅ Choose the right exception type for different errors
- ✅ Create custom exception hierarchies
- ✅ Use RAII for automatic resource management
- ✅ Apply exception safety guarantees
- ✅ Use noexcept appropriately
- ✅ Handle exceptions in constructors and destructors
- ✅ Debug exception-related issues

---

## 📚 Files Overview

```
exception/
├── README.md                      # Detailed guide
├── QUICK_START.md                 # This file
├── Makefile                       # Build automation
├── 01_basic_exception.cpp         # Start here!
├── 02_multiple_catch.cpp
├── 03_standard_exceptions.cpp
├── 04_custom_exceptions.cpp
├── 05_rethrow_exceptions.cpp
├── 06_nested_try_catch.cpp
├── 07_constructor_exceptions.cpp
├── 08_exception_safety.cpp
├── 09_noexcept.cpp
└── 10_real_world_example.cpp     # Finish here!
```

---

## 🎓 Start Learning Now!

```bash
# Compile and run the first example
cd "/media/dell/Hard Drive/Summer of code/Robotics/ROS2/ros2-learn/more-cpp/exception/"
g++ -std=c++17 -Wall -Wextra 01_basic_exception.cpp -o build/01_basic_exception
./build/01_basic_exception
```

Then open `01_basic_exception.cpp` in your favorite editor and start reading!

---

## 🔧 Troubleshooting

**Q: Compilation errors?**
A: Make sure you're using C++17: `g++ -std=c++17 ...`

**Q: "No such file or directory"?**
A: Create build directory: `mkdir -p build`

**Q: Want to clean up?**
A: `rm -rf build/ bank_transactions.log`

---

Happy Learning! 🚀
