// Parallelize loops - work distribution across threads
#include <stdio.h>
#include <omp.h>

int main() {
    const int N = 20;

    printf("Sequential loop:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", i);
    }
    printf("\n\n");

    printf("Parallel loop (note the order):\n");
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        printf("%d[t%d] ", i, omp_get_thread_num());
    }
    printf("\n");

    return 0;
}
