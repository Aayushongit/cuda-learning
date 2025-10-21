CC = gcc
CFLAGS = -Wall -Wextra -pthread -std=c11
LDFLAGS = -pthread

SOURCES = $(wildcard *.c)
TARGETS = $(SOURCES:.c=)

.PHONY: all clean run

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

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
