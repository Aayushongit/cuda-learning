#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

void* calculate_square(void* arg) {
    int num = *(int*)arg;
    int* result = malloc(sizeof(int));
    *result = num * num;
    return result;
}

int main() {
    pthread_t thread;
    int number = 7;
    int* result;

    pthread_create(&thread, NULL, calculate_square, &number);
    pthread_join(thread, (void**)&result);

    printf("Square of %d is %d\n", number, *result);
    free(result);

    return 0;
}
