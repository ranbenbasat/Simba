# Makefile for Simba C++ project
# Supports Linux and macOS compilation

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I.

# Source files
SOURCES = main.cpp Simba.cpp defs.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = simba

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (for development)
install-deps:
	@echo "Installing dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		echo "Detected Ubuntu/Debian, installing build-essential..."; \
		sudo apt-get update && sudo apt-get install -y build-essential; \
	elif command -v brew >/dev/null 2>&1; then \
		echo "Detected macOS, installing Xcode command line tools..."; \
		xcode-select --install; \
	else \
		echo "Please install C++ compiler manually"; \
	fi

# Run the program
run: $(TARGET)
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the project (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  run          - Build and run the program"
	@echo "  install-deps - Install system dependencies"
	@echo "  help         - Show this help message"

.PHONY: all clean run install-deps help
