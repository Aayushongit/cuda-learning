#include <stdio.h>
#include <pthread.h>

pthread_spinlock_t spinlock;
int counter = 0;

void* increment_with_spinlock(void* arg) {
    for (int i = 0; i < 100000; i++) {
        pthread_spin_lock(&spinlock);
        counter++;
        pthread_spin_unlock(&spinlock);
    }
    return NULL;
}

int main() {
    pthread_t threads[4];

    pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);

    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, increment_with_spinlock, NULL);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Final counter: %d (expected: 400000)\n", counter);

    pthread_spin_destroy(&spinlock);
    return 0;
}
