#include <stdio.h>
#include <pthread.h>

void* print_number(void* arg) {
    int num = *(int*)arg;
    printf("Thread received: %d\n", num);
    printf("Thread ID: %lu\n", pthread_self());
    return NULL;
}

int main() {
    pthread_t threads[3];
    int numbers[3] = {10, 20, 30};

    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, print_number, &numbers[i]);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
