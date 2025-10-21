// Different loop scheduling strategies
#include <stdio.h>
#include <omp.h>

void print_iteration_assignment(int i) {
    printf("Iter %2d -> Thread %d\n", i, omp_get_thread_num());
}

int main() {
    const int N = 16;
    omp_set_num_threads(4);

    printf("STATIC scheduling (default - contiguous chunks):\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        print_iteration_assignment(i);
    }

    printf("\nSTATIC with chunk size 2:\n");
    #pragma omp parallel for schedule(static, 2)
    for (int i = 0; i < N; i++) {
        print_iteration_assignment(i);
    }

    printf("\nDYNAMIC scheduling (dynamic assignment):\n");
    #pragma omp parallel for schedule(dynamic, 2)
    for (int i = 0; i < N; i++) {
        print_iteration_assignment(i);
    }

    printf("\nGUIDED scheduling (decreasing chunk sizes):\n");
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        print_iteration_assignment(i);
    }

    return 0;
}
