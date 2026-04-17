MPICC ?= mpicc
CFLAGS ?= -O3 -Wall -Wextra -std=c11

SRC_DIR := src
BIN_DIR := bin

PROGRAMS := hello_mpi send_recv broadcast reduce_sum matrix_mult ring_comm array_sum max_min dot_product perf_metrics
TARGETS := $(addprefix $(BIN_DIR)/,$(PROGRAMS))

NP ?= 4
PROG ?= hello_mpi
ARGS ?=

.PHONY: all clean run run-examples run-exercises perf-help

all: $(BIN_DIR) $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(SRC_DIR)/%.c | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $< -o $@

run: all
	mpirun -np $(NP) $(BIN_DIR)/$(PROG) $(ARGS)

run-examples: all
	mpirun -np 4 $(BIN_DIR)/hello_mpi
	mpirun -np 2 $(BIN_DIR)/send_recv
	mpirun -np 4 $(BIN_DIR)/broadcast
	mpirun -np 4 $(BIN_DIR)/reduce_sum
	mpirun -np 4 $(BIN_DIR)/matrix_mult

run-exercises: all
	mpirun -np 4 $(BIN_DIR)/ring_comm
	mpirun -np 4 $(BIN_DIR)/array_sum
	mpirun -np 4 $(BIN_DIR)/max_min
	mpirun -np 4 $(BIN_DIR)/dot_product

perf-help:
	@echo "Run performance metrics with:"
	@echo "  mpirun -np 1 $(BIN_DIR)/perf_metrics 1024"
	@echo "  mpirun -np 2 $(BIN_DIR)/perf_metrics 1024"
	@echo "  mpirun -np 4 $(BIN_DIR)/perf_metrics 1024"
	@echo "  mpirun -np 8 $(BIN_DIR)/perf_metrics 1024"

clean:
	rm -rf $(BIN_DIR)
