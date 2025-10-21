// Complete OpenMP pragma reference with examples
#include <stdio.h>
#include <omp.h>

void section_parallel_regions() {
    printf("1. PARALLEL REGIONS\n");
    printf("-------------------\n");

    // Basic
    #pragma omp parallel
    { printf("  Basic parallel\n"); }

    // With clauses
    #pragma omp parallel num_threads(2) if(1)
    { printf("  With num_threads and if\n"); }

    printf("\n");
}

void section_work_sharing() {
    printf("2. WORK-SHARING CONSTRUCTS\n");
    printf("--------------------------\n");

    // Parallel for
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        printf("  parallel for: %d\n", i);
    }

    // Sections
    #pragma omp parallel sections
    {
        #pragma omp section
        printf("  Section A\n");

        #pragma omp section
        printf("  Section B\n");
    }

    // Single
    #pragma omp parallel
    {
        #pragma omp single
        printf("  Single thread executes this\n");
    }

    printf("\n");
}

void section_scheduling() {
    printf("3. SCHEDULING TYPES\n");
    printf("-------------------\n");

    printf("  schedule(static)      - Fixed chunks\n");
    printf("  schedule(static, N)   - Chunks of size N\n");
    printf("  schedule(dynamic)     - Dynamic assignment\n");
    printf("  schedule(dynamic, N)  - Dynamic with chunk N\n");
    printf("  schedule(guided)      - Decreasing chunk sizes\n");
    printf("  schedule(auto)        - Compiler decides\n");
    printf("  schedule(runtime)     - Use OMP_SCHEDULE env var\n");
    printf("\n");
}

void section_data_clauses() {
    printf("4. DATA SHARING CLAUSES\n");
    printf("-----------------------\n");

    int x = 10;

    #pragma omp parallel private(x) num_threads(2)
    {
        printf("  private(x): each thread has own x (uninitialized)\n");
    }

    #pragma omp parallel firstprivate(x) num_threads(2)
    {
        printf("  firstprivate(x): private but initialized to %d\n", x);
    }

    #pragma omp parallel for lastprivate(x)
    for (int i = 0; i < 2; i++) {
        x = i;
    }
    printf("  lastprivate(x): x = %d (from last iteration)\n", x);

    #pragma omp parallel shared(x) num_threads(2)
    {
        printf("  shared(x): all threads access same x\n");
    }

    printf("\n");
}

void section_synchronization() {
    printf("5. SYNCHRONIZATION\n");
    printf("------------------\n");

    int counter = 0;

    #pragma omp parallel num_threads(2)
    {
        #pragma omp critical
        {
            counter++;
            printf("  critical: only one thread at a time\n");
        }

        #pragma omp atomic
        counter++;
        printf("  atomic: lightweight for simple operations\n");

        #pragma omp barrier
        printf("  barrier: wait for all threads\n");
    }

    printf("\n");
}

void section_reductions() {
    printf("6. REDUCTION OPERATIONS\n");
    printf("-----------------------\n");

    printf("  reduction(+:var)   - Sum\n");
    printf("  reduction(-:var)   - Difference\n");
    printf("  reduction(*:var)   - Product\n");
    printf("  reduction(&:var)   - Bitwise AND\n");
    printf("  reduction(|:var)   - Bitwise OR\n");
    printf("  reduction(^:var)   - Bitwise XOR\n");
    printf("  reduction(&&:var)  - Logical AND\n");
    printf("  reduction(||:var)  - Logical OR\n");
    printf("  reduction(max:var) - Maximum\n");
    printf("  reduction(min:var) - Minimum\n");
    printf("\n");
}

void section_tasks() {
    printf("7. TASKS\n");
    printf("--------\n");

    #pragma omp parallel num_threads(2)
    {
        #pragma omp single
        {
            #pragma omp task
            printf("  task: dynamic work unit\n");

            #pragma omp task priority(10)
            printf("  task priority: hint for scheduling\n");

            #pragma omp taskwait
            printf("  taskwait: wait for child tasks\n");
        }
    }

    printf("\n");
}

void section_loop_modifiers() {
    printf("8. LOOP MODIFIERS\n");
    printf("-----------------\n");

    // Collapse
    #pragma omp parallel for collapse(2) num_threads(2)
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("  collapse(2): combines nested loops\n");
        }
    }

    // Ordered
    #pragma omp parallel for ordered num_threads(2)
    for (int i = 0; i < 2; i++) {
        #pragma omp ordered
        printf("  ordered: sequential execution of block\n");
    }

    printf("\n");
}

void section_conditional() {
    printf("9. CONDITIONAL PARALLELIZATION\n");
    printf("------------------------------\n");

    int n = 100;

    #pragma omp parallel if(n > 50) num_threads(2)
    {
        printf("  if(condition): parallel only if true\n");
    }

    printf("\n");
}

void section_advanced() {
    printf("10. ADVANCED FEATURES\n");
    printf("---------------------\n");

    // Nowait
    #pragma omp parallel num_threads(2)
    {
        #pragma omp for nowait
        for (int i = 0; i < 2; i++) {
            printf("  nowait: skip implicit barrier\n");
        }
    }

    // Master
    #pragma omp parallel num_threads(2)
    {
        #pragma omp master
        printf("  master: only master thread (tid=0)\n");
    }

    // SIMD
    int a[4] = {0};
    #pragma omp simd
    for (int i = 0; i < 4; i++) {
        a[i] = i;
    }
    printf("  simd: vectorization hint\n");

    printf("\n");
}

void print_quick_reference() {
    printf("QUICK REFERENCE CARD\n");
    printf("====================\n\n");

    printf("Basic Structure:\n");
    printf("  #pragma omp parallel          - Create parallel region\n");
    printf("  #pragma omp parallel for      - Parallel loop\n");
    printf("  #pragma omp for               - Work-sharing in parallel region\n\n");

    printf("Common Patterns:\n");
    printf("  #pragma omp parallel for reduction(+:sum)\n");
    printf("  #pragma omp parallel for schedule(dynamic) private(x)\n");
    printf("  #pragma omp parallel for collapse(2) num_threads(4)\n");
    printf("  #pragma omp parallel for if(n > 1000)\n\n");

    printf("Synchronization:\n");
    printf("  #pragma omp critical          - Mutual exclusion\n");
    printf("  #pragma omp atomic           - Atomic operation\n");
    printf("  #pragma omp barrier          - Wait for all threads\n\n");

    printf("Data Sharing:\n");
    printf("  private(x)        - Each thread has own x\n");
    printf("  shared(x)         - All threads share x\n");
    printf("  firstprivate(x)   - Private, initialized\n");
    printf("  lastprivate(x)    - Value from last iteration\n");
    printf("  reduction(op:x)   - Combine results\n\n");

    printf("Tasks:\n");
    printf("  #pragma omp task             - Create task\n");
    printf("  #pragma omp taskwait         - Wait for tasks\n");
    printf("  depend(in:x) depend(out:y)  - Task dependencies\n\n");

    printf("Environment Variables:\n");
    printf("  OMP_NUM_THREADS=8            - Set thread count\n");
    printf("  OMP_SCHEDULE=\"dynamic,16\"    - Set scheduling\n");
    printf("  OMP_PROC_BIND=true           - Thread binding\n\n");
}

int main() {
    printf("OpenMP Pragma Complete Reference\n");
    printf("=================================\n\n");

    section_parallel_regions();
    section_work_sharing();
    section_scheduling();
    section_data_clauses();
    section_synchronization();
    section_reductions();
    section_tasks();
    section_loop_modifiers();
    section_conditional();
    section_advanced();

    printf("\n");
    print_quick_reference();

    return 0;
}
