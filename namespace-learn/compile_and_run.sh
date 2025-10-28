#!/bin/bash

# Compilation script for all namespace examples

echo "=== Compiling and Running C++ Namespace Examples ==="
echo ""

# Function to compile and run
compile_and_run() {
    local source=$1
    local output=$2
    local description=$3

    echo "----------------------------------------"
    echo "Example: $description"
    echo "----------------------------------------"

    if g++ $source -o $output 2>/dev/null; then
        ./$output
        rm $output
    else
        echo "Compilation failed for $source"
    fi
    echo ""
}

# Run all examples
compile_and_run "01_basic_namespace.cpp" "01_program" "Basic Namespace Usage"
compile_and_run "02_nested_namespace.cpp" "02_program" "Nested Namespaces"
compile_and_run "03_multiple_namespaces.cpp" "03_program" "Multiple Namespaces"
compile_and_run "04_namespace_alias.cpp" "04_program" "Namespace Aliases"
compile_and_run "05_using_directive.cpp" "05_program" "Using Directive"
compile_and_run "06_anonymous_namespace.cpp" "06_program" "Anonymous Namespace"

# Example 7 requires multiple files
echo "----------------------------------------"
echo "Example: Namespace Across Multiple Files"
echo "----------------------------------------"
if g++ 07_main.cpp 07_geometry.cpp -o 07_program 2>/dev/null; then
    ./07_program
    rm 07_program
else
    echo "Compilation failed for example 07"
fi
echo ""

compile_and_run "08_namespace_classes.cpp" "08_program" "Classes in Namespaces"
compile_and_run "09_inline_namespace.cpp" "09_program" "Inline Namespaces"
compile_and_run "10_real_world_example.cpp" "10_program" "Real-World Project"

echo "=== All examples completed! ==="
