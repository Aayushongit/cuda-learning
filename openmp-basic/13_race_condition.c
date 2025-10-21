// Demonstrating race conditions and their fixes
#include <stdio.h>
#include <omp.h>

int main() {
    int balance = 1000;

    printf("Initial balance: %d\n\n", balance);

    // UNSAFE: Race condition
    printf("Running UNSAFE version (race condition):\n");
    balance = 1000;
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 100; i++) {
        int temp = balance;
        temp += 10;
        balance = temp;
    }
    printf("Final balance (unsafe): %d (expected: 2000)\n\n", balance);

    // FIX 1: Critical section
    printf("FIX 1: Using critical section:\n");
    balance = 1000;
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 100; i++) {
        #pragma omp critical
        {
            int temp = balance;
            temp += 10;
            balance = temp;
        }
    }
    printf("Final balance (critical): %d (expected: 2000)\n\n", balance);

    // FIX 2: Atomic (simpler case)
    printf("FIX 2: Using atomic:\n");
    balance = 1000;
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 100; i++) {
        #pragma omp atomic
        balance += 10;
    }
    printf("Final balance (atomic): %d (expected: 2000)\n\n", balance);

    // FIX 3: Reduction
    printf("FIX 3: Using reduction:\n");
    balance = 1000;
    int total_deposit = 0;
    #pragma omp parallel for reduction(+:total_deposit) num_threads(4)
    for (int i = 0; i < 100; i++) {
        total_deposit += 10;
    }
    balance += total_deposit;
    printf("Final balance (reduction): %d (expected: 2000)\n", balance);

    return 0;
}
