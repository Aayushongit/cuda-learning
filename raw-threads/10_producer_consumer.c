#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 5

int buffer[BUFFER_SIZE];
int count = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;

void* producer(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 10; i++) {
        pthread_mutex_lock(&mutex);

        while (count == BUFFER_SIZE) {
            pthread_cond_wait(&not_full, &mutex);
        }

        buffer[count++] = i;
        printf("Producer %d: produced %d (buffer size: %d)\n", id, i, count);

        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&mutex);

        usleep(100000);
    }
    return NULL;
}

void* consumer(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 10; i++) {
        pthread_mutex_lock(&mutex);

        while (count == 0) {
            pthread_cond_wait(&not_empty, &mutex);
        }

        int item = buffer[--count];
        printf("Consumer %d: consumed %d (buffer size: %d)\n", id, item, count);

        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&mutex);

        usleep(150000);
    }
    return NULL;
}

int main() {
    pthread_t prod, cons;
    int prod_id = 1, cons_id = 1;

    pthread_create(&prod, NULL, producer, &prod_id);
    pthread_create(&cons, NULL, consumer, &cons_id);

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&not_full);
    pthread_cond_destroy(&not_empty);

    return 0;
}
