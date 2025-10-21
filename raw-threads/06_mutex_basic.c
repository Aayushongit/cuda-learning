#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int shared_resource = 0;

void* safe_increment(void* arg) {
    for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&mutex);
        shared_resource++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main() {
    pthread_t threads[10];

    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, safe_increment, NULL);
    }

    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Final value: %d (expected: 1000000)\n", shared_resource);
    pthread_mutex_destroy(&mutex);

    return 0;
}
