#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define POOL_SIZE 4
#define QUEUE_SIZE 10

typedef struct {
    void (*function)(int);
    int argument;
} task_t;

task_t task_queue[QUEUE_SIZE];
int queue_head = 0;
int queue_tail = 0;
int queue_count = 0;

pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t queue_cond = PTHREAD_COND_INITIALIZER;
int shutdown = 0;

void execute_task(int arg) {
    printf("Thread %lu: Processing task with arg %d\n", pthread_self(), arg);
    sleep(1);
}

void* thread_pool_worker(void* arg) {
    while (1) {
        pthread_mutex_lock(&queue_mutex);

        while (queue_count == 0 && !shutdown) {
            pthread_cond_wait(&queue_cond, &queue_mutex);
        }

        if (shutdown && queue_count == 0) {
            pthread_mutex_unlock(&queue_mutex);
            break;
        }

        task_t task = task_queue[queue_head];
        queue_head = (queue_head + 1) % QUEUE_SIZE;
        queue_count--;

        pthread_mutex_unlock(&queue_mutex);

        task.function(task.argument);
    }

    return NULL;
}

void add_task(void (*function)(int), int argument) {
    pthread_mutex_lock(&queue_mutex);

    task_queue[queue_tail].function = function;
    task_queue[queue_tail].argument = argument;
    queue_tail = (queue_tail + 1) % QUEUE_SIZE;
    queue_count++;

    pthread_cond_signal(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);
}

int main() {
    pthread_t threads[POOL_SIZE];

    for (int i = 0; i < POOL_SIZE; i++) {
        pthread_create(&threads[i], NULL, thread_pool_worker, NULL);
    }

    for (int i = 0; i < 8; i++) {
        add_task(execute_task, i + 1);
        printf("Main: Added task %d to queue\n", i + 1);
    }

    sleep(5);

    pthread_mutex_lock(&queue_mutex);
    shutdown = 1;
    pthread_cond_broadcast(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);

    for (int i = 0; i < POOL_SIZE; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&queue_mutex);
    pthread_cond_destroy(&queue_cond);

    printf("All tasks completed, thread pool shut down\n");
    return 0;
}
