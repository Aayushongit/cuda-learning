#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define MAX_RESOURCES 3
#define NUM_THREADS 5

sem_t semaphore;

void* use_resource(void* arg) {
    int id = *(int*)arg;

    printf("Thread %d: Waiting for resource\n", id);
    sem_wait(&semaphore);

    printf("Thread %d: Acquired resource\n", id);
    sleep(2);
    printf("Thread %d: Releasing resource\n", id);

    sem_post(&semaphore);

    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    sem_init(&semaphore, 0, MAX_RESOURCES);

    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i + 1;
        pthread_create(&threads[i], NULL, use_resource, &ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    sem_destroy(&semaphore);
    return 0;
}
