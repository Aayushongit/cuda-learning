# Makefile for C++ Exception Handling Tutorial
# Usage:
#   make         - Build all examples
#   make run     - Run all examples in sequence
#   make clean   - Remove all built files
#   make 01      - Build and run example 01
#   make help    - Show available commands

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic
BUILDDIR = build

# Source files
SOURCES = $(wildcard [0-9][0-9]_*.cpp)
# Target executables
TARGETS = $(patsubst %.cpp,$(BUILDDIR)/%,$(basename $(SOURCES)))

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

.PHONY: all clean run help $(basename $(SOURCES))

# Default target
all: $(BUILDDIR) $(TARGETS)
	@echo "$(GREEN)✓ All examples compiled successfully!$(NC)"
	@echo "$(BLUE)Run 'make run' to execute all examples$(NC)"
	@echo "$(BLUE)Or run individual examples: ./build/01_basic_exception$(NC)"

# Create build directory
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)
	@echo "$(BLUE)Created build directory$(NC)"

# Pattern rule to build each example
$(BUILDDIR)/%: %.cpp | $(BUILDDIR)
	@echo "$(YELLOW)Compiling $<...$(NC)"
	@$(CXX) $(CXXFLAGS) $< -o $@
	@echo "$(GREEN)✓ Built $@$(NC)"


# Run all examples in sequence
run: all
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Running all examples...$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@for target in $(TARGETS); do \
		echo ""; \
		echo "$(BLUE)======================================== "; \
		echo "Running $$target"; \
		echo "========================================$(NC)"; \
		./$$target; \
		echo ""; \
		echo "$(YELLOW)Press Enter to continue...$(NC)"; \
		read dummy; \
	done
	@echo "$(GREEN)All examples completed!$(NC)"

# Individual targets for each example
01: $(BUILDDIR)/01_basic_exception
	@echo "$(BLUE)Running 01_basic_exception...$(NC)"
	@./$(BUILDDIR)/01_basic_exception

02: $(BUILDDIR)/02_multiple_catch
	@echo "$(BLUE)Running 02_multiple_catch...$(NC)"
	@./$(BUILDDIR)/02_multiple_catch

03: $(BUILDDIR)/03_standard_exceptions
	@echo "$(BLUE)Running 03_standard_exceptions...$(NC)"
	@./$(BUILDDIR)/03_standard_exceptions

04: $(BUILDDIR)/04_custom_exceptions
	@echo "$(BLUE)Running 04_custom_exceptions...$(NC)"
	@./$(BUILDDIR)/04_custom_exceptions

05: $(BUILDDIR)/05_rethrow_exceptions
	@echo "$(BLUE)Running 05_rethrow_exceptions...$(NC)"
	@./$(BUILDDIR)/05_rethrow_exceptions

06: $(BUILDDIR)/06_nested_try_catch
	@echo "$(BLUE)Running 06_nested_try_catch...$(NC)"
	@./$(BUILDDIR)/06_nested_try_catch

07: $(BUILDDIR)/07_constructor_exceptions
	@echo "$(BLUE)Running 07_constructor_exceptions...$(NC)"
	@./$(BUILDDIR)/07_constructor_exceptions

08: $(BUILDDIR)/08_exception_safety
	@echo "$(BLUE)Running 08_exception_safety...$(NC)"
	@./$(BUILDDIR)/08_exception_safety

09: $(BUILDDIR)/09_noexcept
	@echo "$(BLUE)Running 09_noexcept...$(NC)"
	@./$(BUILDDIR)/09_noexcept

10: $(BUILDDIR)/10_real_world_example
	@echo "$(BLUE)Running 10_real_world_example...$(NC)"
	@./$(BUILDDIR)/10_real_world_example

# Clean build directory
clean:
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@rm -rf $(BUILDDIR)
	@rm -f bank_transactions.log
	@echo "$(GREEN)✓ Clean complete$(NC)"

# Help target
help:
	@echo "$(GREEN)C++ Exception Handling Tutorial - Makefile Help$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@echo "  $(BLUE)make$(NC)          - Compile all examples"
	@echo "  $(BLUE)make run$(NC)      - Compile and run all examples (with pauses)"
	@echo "  $(BLUE)make clean$(NC)    - Remove all built files"
	@echo "  $(BLUE)make help$(NC)     - Show this help message"
	@echo ""
	@echo "$(YELLOW)Run individual examples:$(NC)"
	@echo "  $(BLUE)make 01$(NC)       - Build and run 01_basic_exception"
	@echo "  $(BLUE)make 02$(NC)       - Build and run 02_multiple_catch"
	@echo "  $(BLUE)make 03$(NC)       - Build and run 03_standard_exceptions"
	@echo "  $(BLUE)make 04$(NC)       - Build and run 04_custom_exceptions"
	@echo "  $(BLUE)make 05$(NC)       - Build and run 05_rethrow_exceptions"
	@echo "  $(BLUE)make 06$(NC)       - Build and run 06_nested_try_catch"
	@echo "  $(BLUE)make 07$(NC)       - Build and run 07_constructor_exceptions"
	@echo "  $(BLUE)make 08$(NC)       - Build and run 08_exception_safety"
	@echo "  $(BLUE)make 09$(NC)       - Build and run 09_noexcept"
	@echo "  $(BLUE)make 10$(NC)       - Build and run 10_real_world_example"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make          # Compile all"
	@echo "  make 01       # Run just the first example"
	@echo "  make run      # Run all examples with pauses between them"
