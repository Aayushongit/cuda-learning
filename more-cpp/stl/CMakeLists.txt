cmake_minimum_required(VERSION 3.25)
project(CPP_STL_Tutorial)

# Set C++ standard to C++20 for modern features
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable threading support
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# List of all example executables
set(EXAMPLES
    01_vector_basics
    02_array_deque
    03_list_forward_list
    04_set_map
    05_unordered_containers
    06_container_adapters
    07_iterators
    08_algorithms_part1
    09_algorithms_part2
    10_lambda_functional
    11_smart_pointers
    12_strings
    13_streams_io
    14_regex
    15_chrono_time
    16_threading
    17_filesystem
    18_optional_variant_any
    19_tuple_pair
    20_exception_handling
    21_templates
    22_type_traits
    23_move_semantics
    24_ranges_views
)

# Create executables for each example
foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} PRIVATE Threads::Threads)

    # Link filesystem library if needed
    if(${EXAMPLE} STREQUAL "17_filesystem")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
            target_link_libraries(${EXAMPLE} PRIVATE stdc++fs)
        endif()
    endif()
endforeach()

# Build all examples target
add_custom_target(build_all_examples
    DEPENDS ${EXAMPLES}
    COMMENT "Building all STL examples"
)

# Print build information
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Building ${CMAKE_PROJECT_NAME} with ${CMAKE_CXX_COMPILER_ID} compiler")
message(STATUS "Number of examples: ${CMAKE_MATCH_COUNT}")
