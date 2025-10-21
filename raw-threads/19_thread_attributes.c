#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void* worker(void* arg) {
    pthread_attr_t attr;
    size_t stack_size;
    int detach_state;

    pthread_getattr_np(pthread_self(), &attr);
    pthread_attr_getstacksize(&attr, &stack_size);
    pthread_attr_getdetachstate(&attr, &detach_state);

    printf("Thread stack size: %zu bytes\n", stack_size);
    printf("Detach state: %s\n",
           detach_state == PTHREAD_CREATE_DETACHED ? "Detached" : "Joinable");

    pthread_attr_destroy(&attr);
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    pthread_attr_t attr;

    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 2 * 1024 * 1024);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    printf("Creating thread with custom attributes:\n");
    pthread_create(&thread1, &attr, worker, NULL);
    pthread_join(thread1, NULL);

    printf("\nCreating thread with default attributes:\n");
    pthread_create(&thread2, NULL, worker, NULL);
    pthread_join(thread2, NULL);

    pthread_attr_destroy(&attr);
    return 0;
}
