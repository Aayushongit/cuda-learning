#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void* detached_worker(void* arg) {
    int id = *(int*)arg;
    printf("Detached thread %d starting\n", id);
    sleep(2);
    printf("Detached thread %d finishing\n", id);
    return NULL;
}

void* joined_worker(void* arg) {
    int id = *(int*)arg;
    printf("Joined thread %d starting\n", id);
    sleep(1);
    printf("Joined thread %d finishing\n", id);
    return NULL;
}

int main() {
    pthread_t detached_thread, joined_thread;
    int id1 = 1, id2 = 2;

    pthread_create(&detached_thread, NULL, detached_worker, &id1);
    pthread_detach(detached_thread);

    pthread_create(&joined_thread, NULL, joined_worker, &id2);

    pthread_join(joined_thread, NULL);
    printf("Main: Joined thread completed\n");

    sleep(3);
    printf("Main: Exiting (detached thread runs independently)\n");

    return 0;
}
