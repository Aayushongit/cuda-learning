// Sections - divide different tasks among threads
#include <stdio.h>
#include <omp.h>

void task_a() {
    printf("Task A executed by thread %d\n", omp_get_thread_num());
}

void task_b() {
    printf("Task B executed by thread %d\n", omp_get_thread_num());
}

void task_c() {
    printf("Task C executed by thread %d\n", omp_get_thread_num());
}

void task_d() {
    printf("Task D executed by thread %d\n", omp_get_thread_num());
}

int main() {
    omp_set_num_threads(4);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            task_a();
        }

        #pragma omp section
        {
            task_b();
        }

        #pragma omp section
        {
            task_c();
        }

        #pragma omp section
        {
            task_d();
        }
    }

    return 0;
}
