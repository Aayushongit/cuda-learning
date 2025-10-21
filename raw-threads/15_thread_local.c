#include <stdio.h>
#include <pthread.h>

pthread_key_t key;

void destructor(void* value) {
    printf("Destructor called for value: %d\n", *(int*)value);
    free(value);
}

void* worker(void* arg) {
    int id = *(int*)arg;

    int* local_data = malloc(sizeof(int));
    *local_data = id * 100;

    pthread_setspecific(key, local_data);

    printf("Thread %d: Set thread-local value to %d\n", id, *local_data);

    int* retrieved = (int*)pthread_getspecific(key);
    printf("Thread %d: Retrieved thread-local value: %d\n", id, *retrieved);

    return NULL;
}

int main() {
    pthread_t threads[3];
    int ids[] = {1, 2, 3};

    pthread_key_create(&key, destructor);

    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, worker, &ids[i]);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_key_delete(key);
    return 0;
}
