// Reduction - efficient way to combine results from all threads
#include <stdio.h>
#include <omp.h>

int main() {
    const int N = 1000000;

    // Sum with reduction clause
    long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= N; i++) {
        sum += i;
    }
    printf("Sum 1 to %d = %ld\n", N, sum);

    // Multiple reduction operations
    int max_val = 0, min_val = N;
    long product = 1;

    #pragma omp parallel for reduction(max:max_val) reduction(min:min_val)
    for (int i = 1; i <= 100; i++) {
        if (i > max_val) max_val = i;
        if (i < min_val) min_val = i;
    }

    printf("Max: %d, Min: %d\n", max_val, min_val);

    // Custom reduction with array
    int arr[10] = {0};
    #pragma omp parallel for
    for (int i = 0; i < 1000; i++) {
        int idx = i % 10;
        #pragma omp atomic
        arr[idx] += i;
    }

    printf("Array sums: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
