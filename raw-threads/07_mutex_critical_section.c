#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* print_pattern(void* arg) {
    char* pattern = (char*)arg;

    pthread_mutex_lock(&mutex);

    for (int i = 0; i < 5; i++) {
        printf("%s", pattern);
        fflush(stdout);
        usleep(100000);
    }
    printf("\n");

    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main() {
    pthread_t t1, t2, t3;

    pthread_create(&t1, NULL, print_pattern, "A");
    pthread_create(&t2, NULL, print_pattern, "B");
    pthread_create(&t3, NULL, print_pattern, "C");

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);

    pthread_mutex_destroy(&mutex);
    return 0;
}
