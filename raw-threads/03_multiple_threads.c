#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define NUM_THREADS 5

typedef struct {
    int thread_id;
    char message[50];
} thread_data_t;

void* worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    printf("Thread %d: %s\n", data->thread_id, data->message);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        snprintf(thread_data[i].message, 50, "Processing task %d", i);
        pthread_create(&threads[i], NULL, worker, &thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads completed\n");
    return 0;
}
