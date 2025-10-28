cmake_minimum_required(VERSION 3.25)
project(CppTypesTemplatesDemo)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executables for each example
add_executable(typedef_demo 01_typedef_examples.cpp)
add_executable(using_demo 02_using_examples.cpp)
add_executable(struct_demo 03_struct_examples.cpp)
add_executable(function_templates_demo 04_function_templates.cpp)
add_executable(class_templates_demo 05_class_templates.cpp)
add_executable(specialization_demo 06_template_specialization.cpp)
add_executable(combined_demo 07_combining_all.cpp)
add_executable(enum_demo 08_enum_examples.cpp)
