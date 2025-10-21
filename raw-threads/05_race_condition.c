#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define INCREMENTS 100000

int counter = 0;

void* increment_counter(void* arg) {
    for (int i = 0; i < INCREMENTS; i++) {
        counter++;  // NOT thread-safe!
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, increment_counter, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Expected: %d\n", NUM_THREADS * INCREMENTS);
    printf("Actual:   %d\n", counter);
    printf("Lost:     %d\n", (NUM_THREADS * INCREMENTS) - counter);

    return 0;
}
