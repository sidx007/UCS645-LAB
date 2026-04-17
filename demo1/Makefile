# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp -march=native

# Target executable
TARGET = correlate_lab

# Source files
SOURCES = main.cpp corelate.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Header files
HEADERS = correlate.h

# Default run configuration (override at runtime if needed)
NY ?= 512
NX ?= 512
THREADS ?= 4
MODE ?= all
PERF_EVENTS ?= cycles,instructions,cache-references,cache-misses

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run benchmark (CLI arguments are configurable from make variables)
run: $(TARGET)
	./$(TARGET) $(NY) $(NX) $(THREADS) $(MODE)

# Linux perf statistics for sequential and parallel modes
perf-seq: $(TARGET)
	perf stat -e $(PERF_EVENTS) ./$(TARGET) $(NY) $(NX) 1 seq

perf-par: $(TARGET)
	perf stat -e $(PERF_EVENTS) ./$(TARGET) $(NY) $(NX) $(THREADS) par

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean run perf-seq perf-par