# C++ Multithreading Tutorial

A comprehensive collection of C++ multithreading examples covering all major concepts, patterns, and algorithms.

## Prerequisites

- C++17 or higher (some examples require C++20)
- Compiler with thread support (g++, clang++, MSVC)
- Basic understanding of C++ programming

## Compilation

Compile any example with:
```bash
g++ -std=c++20 -pthread filename.cpp -o output
./output
```

For older compilers (C++17):
```bash
g++ -std=c++17 -pthread filename.cpp -o output
```

Note: Files 10, 17 require C++20 for semaphore, barrier, and latch support.

## Learning Path

Follow this order to build understanding from basics to advanced concepts:

### 1. Fundamentals (Start Here)

**01_basic_threads.cpp**
- Thread creation and joining
- Passing arguments to threads
- Lambda threads
- Your first introduction to threads

**02_thread_id_hardware.cpp**
- Thread IDs
- Hardware concurrency detection
- Understanding thread capabilities

### 2. Understanding Problems

**03_race_condition.cpp**
- What happens without synchronization
- Demonstrates race conditions
- Why we need thread safety
- IMPORTANT: Run this to understand the problem before learning solutions

### 3. Basic Synchronization

**04_mutex_basics.cpp**
- std::mutex introduction
- lock() and unlock()
- std::lock_guard (RAII pattern)
- Solving race conditions

**05_lock_types.cpp**
- std::lock_guard vs std::unique_lock
- std::scoped_lock (C++17)
- Deferred locking
- try_lock operations

### 4. Advanced Synchronization

**06_condition_variable.cpp**
- Thread coordination
- wait() and notify()
- Timeout operations
- Producer-consumer communication basics

**07_producer_consumer.cpp**
- Classic multithreading pattern
- Queue-based communication
- Multiple producers and consumers
- Real-world application pattern

**08_atomic_operations.cpp**
- Lock-free programming
- std::atomic types
- fetch_add, exchange, compare_exchange
- Memory ordering basics

### 5. Common Problems and Solutions

**09_deadlock.cpp**
- What is deadlock
- How deadlock occurs
- Prevention strategies
- std::lock and std::scoped_lock solutions

**11_shared_mutex.cpp**
- Reader-writer locks
- Multiple readers, single writer
- std::shared_lock
- Thread-safe cache example

### 6. Modern C++20 Features

**10_semaphore.cpp** (C++20)
- std::counting_semaphore
- std::binary_semaphore
- Resource pool management

**17_barrier_latch.cpp** (C++20)
- std::barrier for reusable synchronization
- std::latch for one-time synchronization
- Parallel iteration patterns

### 7. Asynchronous Programming

**12_async_future.cpp**
- std::async for task-based parallelism
- std::future for result retrieval
- Launch policies
- Exception handling across threads

**13_promise.cpp**
- std::promise and std::future
- Manual result setting
- Exception propagation
- Thread synchronization with promises

**18_packaged_task.cpp**
- std::packaged_task
- Task queues
- Deferred execution
- Task reuse

### 8. Advanced Patterns

**14_thread_pool.cpp**
- Thread pool implementation
- Task queue management
- Worker thread coordination
- Production-ready pattern

**16_thread_local.cpp**
- Thread-local storage
- Per-thread variables
- Thread-safe random generators
- Resource management

### 9. Parallel Algorithms

**15_parallel_algorithms.cpp** (C++17)
- Execution policies
- Parallel std::sort, std::for_each
- std::reduce
- Performance comparison

**19_parallel_merge_sort.cpp**
- Divide-and-conquer with threads
- Parallel merge sort implementation
- Performance analysis
- Depth-limited parallelism

**20_parallel_quicksort.cpp**
- Parallel quicksort
- Comparison with sequential
- STL vs custom implementations

## Key Concepts Summary

### Thread Management
- Creation, joining, detaching
- Thread IDs and hardware info
- Lambda and function objects

### Synchronization Primitives
- **Mutex**: Mutual exclusion lock
- **Lock Guards**: RAII lock management
- **Condition Variables**: Thread notification
- **Atomic Operations**: Lock-free operations
- **Semaphore**: Resource counting (C++20)
- **Barrier/Latch**: Bulk synchronization (C++20)

### Common Patterns
- **Producer-Consumer**: Queue-based communication
- **Reader-Writer**: Shared read, exclusive write
- **Thread Pool**: Reusable worker threads
- **Async Tasks**: Future-based parallelism

### Problems to Understand
- **Race Condition**: Unsynchronized data access
- **Deadlock**: Circular lock dependency
- **Thread Safety**: Protecting shared data

### Performance Techniques
- Parallel algorithms with execution policies
- Divide-and-conquer parallelism
- Lock-free programming with atomics
- Thread pool for task management

## Best Practices

1. **Always use RAII locks** (lock_guard, unique_lock, scoped_lock)
2. **Minimize critical sections** (time spent holding locks)
3. **Avoid deadlock** with consistent lock ordering
4. **Use atomic types** for simple counters
5. **Prefer task-based parallelism** (async) over raw threads
6. **Match thread count** to hardware_concurrency()
7. **Handle exceptions** in threaded code
8. **Use thread_local** for per-thread data

## Common Pitfalls

- Forgetting to join or detach threads
- Not protecting shared data
- Locking in different orders (deadlock)
- Creating too many threads
- Ignoring exceptions in threads
- Using regular variables instead of atomic
- Not using RAII (manual lock/unlock)

## Recommended Study Order

**Beginner**: 01 → 02 → 03 → 04 → 05 → 06
**Intermediate**: 07 → 08 → 09 → 11 → 12 → 13
**Advanced**: 14 → 16 → 18 → 15 → 19 → 20
**Modern C++20**: 10 → 17

## Additional Resources

- C++ Concurrency in Action (Book)
- cppreference.com/w/cpp/thread
- YouTube: CppCon talks on concurrency

## Testing Your Understanding

After completing each section, try:
1. Modifying examples to see different behavior
2. Creating your own multithreaded programs
3. Debugging race conditions
4. Optimizing parallel algorithms

## Troubleshooting

**Compilation Errors**:
- Check C++ standard version (use -std=c++20)
- Ensure -pthread flag is included
- Update compiler if using old version

**Runtime Issues**:
- Verify thread safety of shared data
- Check for deadlocks (program hangs)
- Use thread sanitizer: compile with `-fsanitize=thread`

**Performance Issues**:
- Don't create too many threads
- Profile with tools like perf or valgrind
- Compare parallel vs sequential performance

## License

Educational use - feel free to modify and learn from these examples.

---

Happy Threading! Start with file 01 and work your way through systematically.
