# Modern C++ Threading Examples (std::thread)

20 progressive examples using C++11/C++20 standard library threading.

## Why C++ std::thread over pthreads?

- **Cross-platform**: Works on Windows, Linux, macOS
- **Type-safe**: No void* casting, uses templates
- **RAII**: Automatic resource management
- **Modern**: Integrates with C++ standard library
- **Easier**: Less boilerplate, cleaner syntax

## Examples Overview

| # | Example | C++ Features | Standard |
|---|---------|--------------|----------|
| 01 | basic_thread | std::thread, join() | C++11 |
| 02 | thread_with_args | Thread arguments, get_id() | C++11 |
| 03 | multiple_threads | std::vector, emplace_back | C++11 |
| 04 | thread_return_value | std::async, std::future | C++11 |
| 05 | race_condition | Race condition demo | C++11 |
| 06 | mutex_basic | std::mutex, lock_guard | C++11 |
| 07 | mutex_critical_section | Critical sections | C++11 |
| 08 | deadlock | Deadlock demonstration | C++11 |
| 09 | condition_variable | std::condition_variable | C++11 |
| 10 | producer_consumer | Classic pattern with std::queue | C++11 |
| 11 | thread_detach | Detached threads | C++11 |
| 12 | shared_mutex | std::shared_mutex (readers-writers) | C++17 |
| 13 | latch | std::latch (one-time barrier) | C++20 |
| 14 | semaphore | std::counting_semaphore | C++20 |
| 15 | thread_local | thread_local storage | C++11 |
| 16 | scoped_lock | std::scoped_lock (deadlock-free) | C++17 |
| 17 | jthread | std::jthread (auto-joining) | C++20 |
| 18 | atomic_operations | std::atomic | C++11 |
| 19 | hardware_info | hardware_concurrency() | C++11 |
| 20 | thread_pool | Modern thread pool class | C++11 |

## Quick Start

```bash
# Compile all examples
make all

# Run specific example
make run-01_basic_thread

# Or compile manually
g++ -std=c++20 -pthread 01_basic_thread.cpp -o 01_basic_thread
./01_basic_thread

# Clean binaries
make clean
```

## Requirements

- **C++11**: Examples 01-11, 15, 18-20
- **C++17**: Examples 12, 16
- **C++20**: Examples 13, 14, 17

**Compilers:**
- GCC 10+ or Clang 10+ recommended
- For C++20 features: `g++ -std=c++20` or `clang++ -std=c++20`

## Learning Path

**Beginner (C++11 basics):**
1. 01-04: Thread creation, arguments, return values
2. 05-08: Race conditions, mutex, deadlock
3. 09-11: Condition variables, producer-consumer, detachment

**Intermediate (C++17):**
4. 12: Shared mutex for readers-writers
5. 16: Scoped lock for deadlock prevention

**Advanced (C++20):**
6. 13-14: Latch and semaphores
7. 17: jthread with cooperative cancellation
8. 18-20: Atomics, hardware info, thread pool

## Key C++ Threading Components

### Thread Management
- `std::thread` - Basic thread
- `std::jthread` - Auto-joining thread (C++20)
- `thread::hardware_concurrency()` - Get CPU core count

### Synchronization Primitives
- `std::mutex` - Mutual exclusion
- `std::shared_mutex` - Readers-writer lock
- `std::condition_variable` - Thread signaling
- `std::latch` - One-time synchronization point
- `std::counting_semaphore` - Resource limiting

### Lock Management (RAII)
- `std::lock_guard` - Simple RAII lock
- `std::unique_lock` - Flexible RAII lock
- `std::shared_lock` - RAII for shared ownership
- `std::scoped_lock` - Multi-mutex RAII (deadlock-free)

### Asynchronous Operations
- `std::async` - Launch async tasks
- `std::future` - Get async results
- `std::promise` - Set async results

### Atomic Operations
- `std::atomic<T>` - Lock-free atomic types
- Memory ordering options

### Thread-Local Storage
- `thread_local` keyword

## Common Patterns

**Pattern 1: Simple parallel task**
```cpp
std::thread t([]{ /* work */ });
t.join();
```

**Pattern 2: Thread pool**
```cpp
ThreadPool pool(4);
pool.enqueue([]{ /* task */ });
```

**Pattern 3: Producer-Consumer**
```cpp
std::mutex mtx;
std::condition_variable cv;
std::queue<int> queue;
```

**Pattern 4: Async with result**
```cpp
auto future = std::async(std::launch::async, compute);
int result = future.get();
```

## Tips

1. **Always join or detach**: Unjoined threads cause termination
2. **Use RAII locks**: Prefer `lock_guard`/`unique_lock` over manual lock/unlock
3. **Avoid deadlock**: Use `std::scoped_lock` for multiple mutexes
4. **Thread count**: Use `hardware_concurrency()` for CPU-bound tasks
5. **C++20 features**: `jthread` auto-joins, `latch`/`semaphore` simplify sync

## Comparison: pthreads vs std::thread

| Feature | pthreads | C++ std::thread |
|---------|----------|-----------------|
| Platform | POSIX only | Cross-platform |
| Type safety | void* casting | Templates |
| Resource mgmt | Manual | RAII |
| Error handling | Error codes | Exceptions |
| Cancellation | pthread_cancel | std::stop_token |
| Syntax | Verbose | Concise |

## Performance

C++ std::thread is typically a thin wrapper around native threads (pthreads on Linux), so performance is equivalent.

## Further Reading

- C++ reference: https://en.cppreference.com/w/cpp/thread
- C++ Concurrency in Action by Anthony Williams
- Effective Modern C++ by Scott Meyers
