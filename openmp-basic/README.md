# OpenMP Learning Examples

**30 examples**: 22 C examples (fundamentals) + 8 C++ examples (numerical computing)

## Quick Start

```bash
# Compile all examples (C and C++)
make

# Compile only C examples
make c-examples

# Compile only C++ examples
make cpp-examples

# Compile and run a specific example
make run_01_hello_world
make run_cpp_01_vector_ops

# Clean up
make clean
```

## Learning Path

### Basics (Start Here)
1. **01_hello_world.c** - First parallel region, thread creation
2. **02_thread_info.c** - Thread IDs, counts, system info
3. **03_parallel_for.c** - Loop parallelization basics

### Loop Control
4. **04_loop_scheduling.c** - Static, dynamic, guided scheduling strategies
5. **11_nested_loops.c** - Parallelize nested loops, collapse directive

### Data Sharing
6. **05_private_shared.c** - Private vs shared variables
7. **16_firstprivate_lastprivate.c** - Advanced private clauses

### Synchronization
8. **06_critical_section.c** - Mutual exclusion for shared data
9. **07_atomic.c** - Lightweight atomic operations
10. **08_reduction.c** - Efficient parallel reduction operations
11. **09_barrier.c** - Thread synchronization points

### Advanced Parallelism
12. **10_sections.c** - Different tasks on different threads
13. **12_tasks.c** - Dynamic task-based parallelism

### Real-World Problems
14. **13_race_condition.c** - Common pitfalls and solutions
15. **14_matrix_multiply.c** - Matrix multiplication performance
16. **15_pi_calculation.c** - Monte Carlo Pi estimation

### Advanced Directives (Deep Dive)
17. **17_advanced_loop_directives.c** - ordered, if, nowait clauses
18. **18_simd_vectorization.c** - SIMD directives for vectorization
19. **19_advanced_atomic.c** - Atomic read/write/update/capture variants
20. **20_task_dependencies.c** - Task dependencies, priority, taskgroup
21. **21_combined_clauses.c** - Complex examples with multiple clauses
22. **22_pragma_reference.c** - Complete pragma syntax reference

---

## C++ Examples (Numerical Computing Focus)

After mastering C basics, explore these C++ examples for practical numerical computing:

1. **cpp_01_vector_ops.cpp** - Vector operations (dot product, norm, addition)
   - Modern C++ with `std::vector`
   - Reduction operations for dot product
   - Performance timing

2. **cpp_02_matrix_class.cpp** - Object-oriented matrix operations
   - Matrix class with operator overloading
   - Parallel multiplication, addition, transpose
   - Clean C++ interface

3. **cpp_03_numerical_integration.cpp** - Trapezoidal & Simpson's rule
   - Lambda functions with OpenMP
   - Comparing serial vs parallel
   - Multiple integration methods

4. **cpp_04_linear_algebra.cpp** - Linear algebra operations
   - AXPY, matrix-vector product
   - Jacobi iterative solver
   - Real scientific computing patterns

5. **cpp_05_stl_algorithms.cpp** - Parallel STL-style operations
   - Transform, reduce, find_if, count_if
   - Template functions
   - Modern C++ patterns

6. **cpp_06_ode_solver.cpp** - ODE solvers (Euler, RK4)
   - Multiple initial conditions in parallel
   - 2D systems (harmonic oscillator)
   - Scientific simulation

7. **cpp_07_statistics.cpp** - Statistical computing
   - Mean, variance, correlation
   - Parallel histogram computation
   - Data analysis operations

8. **cpp_08_heat_equation.cpp** - PDE solver (2D heat equation)
   - Finite difference method
   - Real physics simulation
   - Multi-dimensional arrays

**Learning Path for C++**: Start with `cpp_01` (vectors), then `cpp_02` (matrices), then choose based on interest:
- Numerical methods: `cpp_03`, `cpp_04`, `cpp_06`, `cpp_08`
- Data/statistics: `cpp_05`, `cpp_07`

---

## Key Concepts

**Parallel Region**: `#pragma omp parallel` - Creates team of threads

**Work Sharing**: `#pragma omp for` - Distributes loop iterations

**Scheduling**:
- `static` - Predictable chunks
- `dynamic` - Load balancing
- `guided` - Adaptive chunk sizes

**Data Clauses**:
- `shared` - Single copy across all threads
- `private` - Each thread has own copy
- `firstprivate` - Private with initialization
- `lastprivate` - Last iteration value copied out

**Synchronization**:
- `critical` - One thread at a time
- `atomic` - Single operation protection
- `barrier` - Wait for all threads
- `reduction` - Combine results efficiently

**Advanced**:
- `sections` - Task parallelism
- `tasks` - Dynamic workloads
- `collapse` - Multi-level loop parallelization

## Environment Variables

```bash
# Set number of threads
export OMP_NUM_THREADS=4

# Set scheduling type
export OMP_SCHEDULE="dynamic,2"
```

## Common Pitfalls

1. **Race conditions** - Multiple threads accessing shared data without synchronization
2. **Over-synchronization** - Too many critical sections hurt performance
3. **False sharing** - Cache line contention on adjacent variables
4. **Load imbalance** - Uneven work distribution across threads

## Tips

**For C Examples:**
- Start with simple examples (01-03)
- Run each example multiple times to see thread scheduling variations
- Experiment with different thread counts
- Compare serial vs parallel performance
- Watch for race conditions in examples 05, 13

**For C++ Examples:**
- Use C++ examples for real numerical computing applications
- Notice how `std::vector` and classes work with OpenMP
- Lambda functions make parallel code cleaner
- Templates allow generic parallel algorithms
- C++ examples show production-ready patterns

**For Advanced Examples (17-22):**
- Examples 17-22 cover all OpenMP pragma variations
- Use example 22 as a quick reference card
- SIMD (18) requires compiler support (use -O2 or -O3)
- Task dependencies (20) show modern OpenMP patterns
- Combined clauses (21) shows real-world complexity

**General:**
- C examples teach OpenMP fundamentals
- C++ examples show practical applications
- Both use identical OpenMP directives
- Performance patterns apply to both languages
- Use `make list` to see examples organized by category
