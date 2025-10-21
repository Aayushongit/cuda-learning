# Raw Threads (POSIX Threads) Learning Examples

20 progressive examples to learn multithreading in C using pthreads.

## Examples Overview

| # | Example | Concepts Covered |
|---|---------|-----------------|
| 01 | basic_thread | Thread creation, joining |
| 02 | thread_with_args | Passing arguments to threads |
| 03 | multiple_threads | Managing multiple threads, struct arguments |
| 04 | thread_return_value | Retrieving return values from threads |
| 05 | race_condition | Demonstrating race conditions |
| 06 | mutex_basic | Mutex basics, critical sections |
| 07 | mutex_critical_section | Protected critical sections |
| 08 | deadlock | Deadlock demonstration and understanding |
| 09 | condition_variable | Condition variables, wait/signal |
| 10 | producer_consumer | Classic producer-consumer problem |
| 11 | thread_detach | Detached vs joinable threads |
| 12 | rwlock | Read-write locks |
| 13 | barrier | Thread barriers, synchronization |
| 14 | semaphore | Semaphores for resource limiting |
| 15 | thread_local | Thread-local storage |
| 16 | spinlock | Spinlocks vs mutexes |
| 17 | thread_cancel | Thread cancellation, cleanup handlers |
| 18 | atomic_operations | Atomic operations |
| 19 | thread_attributes | Thread attributes, stack size |
| 20 | simple_thread_pool | Basic thread pool implementation |

## Compilation

```bash
# Compile all examples
make all

# Compile specific example
make 01_basic_thread

# Compile and run
make run-01_basic_thread

# Clean all binaries
make clean
```

## Learning Path

**Beginner:** 01-04 (Basic thread operations)
**Synchronization:** 05-08 (Race conditions, mutex, deadlock)
**Advanced Sync:** 09-14 (Condition variables, barriers, semaphores)
**Specialized:** 15-19 (Thread-local, spinlocks, atomics)
**Practical:** 20 (Thread pool)

## Notes

- Compile with `-pthread` flag
- Examples use POSIX threads (pthreads)
- Some examples demonstrate anti-patterns (race conditions, deadlock)
- Run examples multiple times to observe non-deterministic behavior
