# Makefile for LAB1 - OpenMP Programs (C++)

CXX = g++
CXXFLAGS = -fopenmp -Wall -O2 -std=c++11

# Example programs
EXAMPLES = eg1_hello_world eg2_num_threads eg3_parallel_for eg4_reduction \
           eg5_execution_time eg6_scheduling eg7_critical_section eg8_private_shared

# Assignment questions
QUESTIONS = q1_daxpy q2_matrix_multiply q3_pi_calculation

# All targets
ALL = $(EXAMPLES) $(QUESTIONS)

.PHONY: all clean examples questions

all: $(ALL)

examples: $(EXAMPLES)

questions: $(QUESTIONS)

# Pattern rule for simple examples
eg1_hello_world: eg1_hello_world.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

eg2_num_threads: eg2_num_threads.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

eg3_parallel_for: eg3_parallel_for.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

eg4_reduction: eg4_reduction.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

eg5_execution_time: eg5_execution_time.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

eg6_scheduling: eg6_scheduling.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

eg7_critical_section: eg7_critical_section.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

eg8_private_shared: eg8_private_shared.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Assignment questions
q1_daxpy: q1_daxpy.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

q2_matrix_multiply: q2_matrix_multiply.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

q3_pi_calculation: q3_pi_calculation.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(ALL)

# Run all examples
run-examples: $(EXAMPLES)
	@echo "=== Running eg1_hello_world ==="
	./eg1_hello_world
	@echo "\n=== Running eg2_num_threads ==="
	./eg2_num_threads
	@echo "\n=== Running eg3_parallel_for ==="
	./eg3_parallel_for
	@echo "\n=== Running eg4_reduction ==="
	./eg4_reduction
	@echo "\n=== Running eg5_execution_time ==="
	./eg5_execution_time
	@echo "\n=== Running eg6_scheduling ==="
	./eg6_scheduling
	@echo "\n=== Running eg7_critical_section ==="
	./eg7_critical_section
	@echo "\n=== Running eg8_private_shared ==="
	./eg8_private_shared

# Run assignment questions
run-questions: $(QUESTIONS)
	@echo "=== Running Q1: DAXPY ==="
	./q1_daxpy
	@echo "\n=== Running Q2: Matrix Multiply ==="
	./q2_matrix_multiply
	@echo "\n=== Running Q3: Pi Calculation ==="
	./q3_pi_calculation
