// Tasks - dynamic parallelism for irregular workloads
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

int fibonacci(int n) {
    if (n < 2) return n;

    int x, y;

    #pragma omp task shared(x)
    x = fibonacci(n - 1);

    #pragma omp task shared(y)
    y = fibonacci(n - 2);

    #pragma omp taskwait
    return x + y;
}

void process_item(int id) {
    printf("Processing item %d on thread %d\n", id, omp_get_thread_num());
    usleep(100000);  // Simulate work
}

int main() {
    omp_set_num_threads(4);

    printf("Task-based processing:\n");
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < 8; i++) {
                #pragma omp task
                {
                    process_item(i);
                }
            }
        }
    }

    printf("\nFibonacci using tasks:\n");
    int result;
    #pragma omp parallel
    {
        #pragma omp single
        {
            result = fibonacci(10);
        }
    }
    printf("Fibonacci(10) = %d\n", result);

    return 0;
}
