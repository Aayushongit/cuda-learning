// Task dependencies and advanced task features
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

void task_work(int id, int duration_ms) {
    printf("  Task %d starting (thread %d)\n", id, omp_get_thread_num());
    usleep(duration_ms * 1000);
    printf("  Task %d completed (thread %d)\n", id, omp_get_thread_num());
}

void demo_basic_task_dependencies() {
    printf("Basic task dependencies (depend clause):\n");

    int x = 0, y = 0, z = 0;

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            // Task A: writes to x
            #pragma omp task depend(out: x) shared(x)
            {
                printf("Task A: computing x\n");
                usleep(100000);
                x = 10;
                printf("Task A: x = %d\n", x);
            }

            // Task B: reads x, writes y (must wait for A)
            #pragma omp task depend(in: x) depend(out: y) shared(x, y)
            {
                printf("Task B: computing y (needs x)\n");
                usleep(100000);
                y = x + 5;
                printf("Task B: y = %d\n", y);
            }

            // Task C: reads y, writes z (must wait for B)
            #pragma omp task depend(in: y) depend(out: z) shared(y, z)
            {
                printf("Task C: computing z (needs y)\n");
                usleep(100000);
                z = y * 2;
                printf("Task C: z = %d\n", z);
            }

            // Task D: independent, can run anytime
            #pragma omp task
            {
                printf("Task D: independent task running\n");
                usleep(50000);
                printf("Task D: completed\n");
            }

            #pragma omp taskwait
            printf("Final: x=%d, y=%d, z=%d\n\n", x, y, z);
        }
    }
}

void demo_parallel_pipeline() {
    printf("Task pipeline with dependencies:\n");

    int data[5] = {0};

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int i = 0; i < 5; i++) {
                // Stage 1: Read
                #pragma omp task depend(out: data[i]) shared(data)
                {
                    printf("  Stage 1: Reading item %d\n", i);
                    usleep(50000);
                    data[i] = i * 10;
                }

                // Stage 2: Process (depends on stage 1)
                #pragma omp task depend(inout: data[i]) shared(data)
                {
                    printf("  Stage 2: Processing item %d (value=%d)\n", i, data[i]);
                    usleep(50000);
                    data[i] = data[i] + 5;
                }

                // Stage 3: Write (depends on stage 2)
                #pragma omp task depend(in: data[i]) shared(data)
                {
                    printf("  Stage 3: Writing item %d (final=%d)\n", i, data[i]);
                    usleep(50000);
                }
            }

            #pragma omp taskwait
            printf("Pipeline completed\n\n");
        }
    }
}

void demo_task_priority() {
    printf("Task priority:\n");

    #pragma omp parallel num_threads(2)
    {
        #pragma omp single
        {
            // Low priority task
            #pragma omp task priority(0)
            {
                task_work(1, 100);
            }

            // High priority task (likely executes first)
            #pragma omp task priority(10)
            {
                task_work(2, 100);
            }

            // Medium priority
            #pragma omp task priority(5)
            {
                task_work(3, 100);
            }

            #pragma omp taskwait
            printf("All priority tasks completed\n\n");
        }
    }
}

void demo_task_final() {
    printf("Task final clause (no more task creation):\n");

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            #pragma omp task final(1)
            {
                printf("  Final task: this and nested tasks run serially\n");

                // This won't create a new task, will execute immediately
                #pragma omp task
                {
                    printf("  Nested task: runs immediately (not deferred)\n");
                }
            }

            #pragma omp taskwait
        }
    }
    printf("\n");
}

void demo_taskgroup() {
    printf("Task groups:\n");

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            printf("Creating task group...\n");

            #pragma omp taskgroup
            {
                for (int i = 0; i < 3; i++) {
                    #pragma omp task
                    {
                        printf("  Outer task %d\n", i);

                        // Create nested tasks
                        #pragma omp task
                        {
                            printf("    Nested task %d.1\n", i);
                        }

                        #pragma omp task
                        {
                            printf("    Nested task %d.2\n", i);
                        }
                    }
                }
                // taskgroup waits for all tasks AND their descendants
            }

            printf("Task group completed (all tasks and descendants done)\n\n");
        }
    }
}

void demo_multidepend() {
    printf("Multiple dependencies:\n");

    int a = 1, b = 2, c = 0, d = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task that depends on multiple inputs
            #pragma omp task depend(in: a, b) depend(out: c) shared(a, b, c)
            {
                printf("  Task computing c = a + b\n");
                usleep(100000);
                c = a + b;
                printf("  c = %d\n", c);
            }

            // Task that produces multiple outputs
            #pragma omp task depend(in: c) depend(out: a, b) shared(a, b, c)
            {
                printf("  Task updating a and b based on c\n");
                usleep(100000);
                a = c * 2;
                b = c * 3;
                printf("  a = %d, b = %d\n", a, b);
            }

            // Task with both in and out on same variable (inout)
            #pragma omp task depend(inout: a) shared(a)
            {
                printf("  Task modifying a in-place\n");
                usleep(100000);
                a += 10;
                printf("  a = %d\n", a);
            }

            #pragma omp taskwait
            printf("Final values: a=%d, b=%d, c=%d\n\n", a, b, c);
        }
    }
}

int main() {
    printf("Task Dependencies and Advanced Features\n");
    printf("========================================\n\n");

    demo_basic_task_dependencies();
    demo_parallel_pipeline();
    demo_task_priority();
    demo_task_final();
    demo_taskgroup();
    demo_multidepend();

    printf("Summary:\n");
    printf("  - depend(in:x)    - task reads x\n");
    printf("  - depend(out:x)   - task writes x\n");
    printf("  - depend(inout:x) - task reads and writes x\n");
    printf("  - priority(n)     - hint for task scheduling\n");
    printf("  - final(1)        - prevent further task creation\n");
    printf("  - taskgroup       - wait for tasks and descendants\n");

    return 0;
}
