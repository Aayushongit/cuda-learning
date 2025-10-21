#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
int shared_data = 0;

void* reader(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 3; i++) {
        pthread_rwlock_rdlock(&rwlock);

        printf("Reader %d: reading value %d\n", id, shared_data);
        usleep(500000);

        pthread_rwlock_unlock(&rwlock);
        usleep(200000);
    }
    return NULL;
}

void* writer(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 3; i++) {
        pthread_rwlock_wrlock(&rwlock);

        shared_data++;
        printf("Writer %d: wrote value %d\n", id, shared_data);
        usleep(700000);

        pthread_rwlock_unlock(&rwlock);
        usleep(300000);
    }
    return NULL;
}

int main() {
    pthread_t readers[3], writers[2];
    int ids[] = {1, 2, 3, 4, 5};

    for (int i = 0; i < 3; i++) {
        pthread_create(&readers[i], NULL, reader, &ids[i]);
    }

    for (int i = 0; i < 2; i++) {
        pthread_create(&writers[i], NULL, writer, &ids[i + 3]);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(readers[i], NULL);
    }
    for (int i = 0; i < 2; i++) {
        pthread_join(writers[i], NULL);
    }

    pthread_rwlock_destroy(&rwlock);
    return 0;
}
