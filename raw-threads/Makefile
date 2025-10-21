CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -pthread
LDFLAGS = -pthread

SOURCES = $(wildcard *.cpp)
TARGETS = $(SOURCES:.cpp=)

.PHONY: all clean help

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)

run-%: %
	@echo "========== Running $< =========="
	@./$<
	@echo ""

help:
	@echo "Available targets:"
	@echo "  make all          - Compile all examples"
	@echo "  make clean        - Remove all compiled binaries"
	@echo "  make <filename>   - Compile specific example (e.g., make 01_basic_thread)"
	@echo "  make run-<name>   - Compile and run specific example"
	@echo ""
	@echo "Examples:"
	@echo "  make 01_basic_thread"
	@echo "  make run-01_basic_thread"
	@echo ""
	@echo "Note: Requires C++20 support (g++ 10+ or clang++ 10+)"
	@echo "      Examples 13, 14, 17 use C++20 features (latch, semaphore, jthread)"
