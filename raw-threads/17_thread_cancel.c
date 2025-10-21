#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void cleanup_handler(void* arg) {
    printf("Cleanup: Releasing resources\n");
}

void* cancelable_worker(void* arg) {
    pthread_cleanup_push(cleanup_handler, NULL);

    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);

    for (int i = 0; i < 10; i++) {
        printf("Worker: Iteration %d\n", i);
        sleep(1);

        pthread_testcancel();
    }

    pthread_cleanup_pop(1);
    return NULL;
}

int main() {
    pthread_t thread;

    pthread_create(&thread, NULL, cancelable_worker, NULL);

    sleep(3);
    printf("Main: Canceling thread\n");
    pthread_cancel(thread);

    pthread_join(thread, NULL);
    printf("Main: Thread canceled and joined\n");

    return 0;
}
