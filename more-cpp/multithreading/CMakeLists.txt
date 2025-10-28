cmake_minimum_required(VERSION 3.10)
project(CPP_Multithreading_Tutorial)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find threads package
find_package(Threads REQUIRED)

# Enable additional warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# List of all example files
set(EXAMPLES
    01_basic_threads
    02_thread_id_hardware
    03_race_condition
    04_mutex_basics
    05_lock_types
    06_condition_variable
    07_producer_consumer
    08_atomic_operations
    09_deadlock
    10_semaphore
    11_shared_mutex
    12_async_future
    13_promise
    14_thread_pool
    15_parallel_algorithms
    16_thread_local
    17_barrier_latch
    18_packaged_task
    19_parallel_merge_sort
    20_parallel_quicksort
)

# Create executable for each example
foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} Threads::Threads)

    # Add TBB for parallel algorithms if available
    if(EXAMPLE STREQUAL "15_parallel_algorithms")
        find_package(TBB QUIET)
        if(TBB_FOUND)
            target_link_libraries(${EXAMPLE} TBB::tbb)
        endif()
    endif()
endforeach()

# Custom target to build all examples
add_custom_target(build_all
    DEPENDS ${EXAMPLES}
    COMMENT "Building all multithreading examples"
)

# Print build information
list(LENGTH EXAMPLES NUM_EXAMPLES)
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Number of examples: ${NUM_EXAMPLES}")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  cmake -B build")
message(STATUS "  cmake --build build")
message(STATUS "  ./build/01_basic_threads")
message(STATUS "")
message(STATUS "Or build specific example:")
message(STATUS "  cmake --build build --target 01_basic_threads")
