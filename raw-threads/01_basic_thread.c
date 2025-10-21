#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void* thread_function(void* arg) {
    printf("Hello from thread!\n");
    sleep(1);
    printf("Thread finishing...\n");
    return NULL;
}

int main() {
    pthread_t thread;

    printf("Main: Creating thread\n");
    pthread_create(&thread, NULL, thread_function, NULL);

    printf("Main: Waiting for thread to finish\n");
    pthread_join(thread, NULL);

    printf("Main: Thread completed\n");
    return 0;
}
