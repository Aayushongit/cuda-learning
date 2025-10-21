#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

#define NUM_THREADS 10
#define INCREMENTS 100000

atomic_int atomic_counter = 0;
int normal_counter = 0;

void* atomic_increment(void* arg) {
    for (int i = 0; i < INCREMENTS; i++) {
        atomic_fetch_add(&atomic_counter, 1);
        normal_counter++;
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, atomic_increment, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    int expected = NUM_THREADS * INCREMENTS;
    printf("Expected:        %d\n", expected);
    printf("Atomic counter:  %d (correct)\n", atomic_counter);
    printf("Normal counter:  %d (race condition)\n", normal_counter);

    return 0;
}
