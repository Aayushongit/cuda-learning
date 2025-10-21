CC = gcc
CXX = g++
CFLAGS = -fopenmp -O2 -Wall
CXXFLAGS = -fopenmp -O2 -Wall -std=c++17
LDFLAGS = -lm

C_SOURCES = $(wildcard [0-9]*.c)
CPP_SOURCES = $(wildcard cpp_*.cpp)
C_EXECUTABLES = $(C_SOURCES:.c=)
CPP_EXECUTABLES = $(CPP_SOURCES:.cpp=)
ALL_EXECUTABLES = $(C_EXECUTABLES) $(CPP_EXECUTABLES)

.PHONY: all clean run c-examples cpp-examples help

all: $(ALL_EXECUTABLES)

c-examples: $(C_EXECUTABLES)

cpp-examples: $(CPP_EXECUTABLES)

# C compilation
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# C++ compilation
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run_%: %
	@echo "=========================================="
	@echo "Running $<"
	@echo "=========================================="
	@./$<
	@echo ""

clean:
	rm -f $(ALL_EXECUTABLES)

help:
	@echo "OpenMP Examples Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make                    - Compile all examples (C and C++)"
	@echo "  make c-examples         - Compile only C examples"
	@echo "  make cpp-examples       - Compile only C++ examples"
	@echo "  make 01_hello_world     - Compile specific example"
	@echo "  make run_01_hello_world - Compile and run example"
	@echo "  make clean              - Remove all executables"
	@echo ""
	@echo "C Examples:"
	@for file in $(C_SOURCES); do echo "  - $${file%.c}"; done
	@echo ""
	@echo "C++ Examples (Numerical Computing):"
	@for file in $(CPP_SOURCES); do echo "  - $${file%.cpp}"; done
