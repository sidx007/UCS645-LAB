MPICC ?= mpicc
CFLAGS ?= -O3 -Wall -Wextra -std=c11
LDLIBS ?= -lm

SRC_DIR := src
BIN_DIR := bin

REQUIRED_PROGRAMS := q1_daxpy q2_broadcast_race q3_distributed_dot q4_primes_master_slave q5_perfect_master_slave
DEMO_PROGRAMS := demo_blocking_ring demo_nonblocking_ring demo_pi_bcast_reduce
PROGRAMS := $(REQUIRED_PROGRAMS) $(DEMO_PROGRAMS)
TARGETS := $(addprefix $(BIN_DIR)/,$(PROGRAMS))

NP ?= 4
PROG ?= q1_daxpy
ARGS ?=

.PHONY: all required demos clean run run-required run-demos help

all: $(BIN_DIR) $(TARGETS)

required: $(BIN_DIR) $(addprefix $(BIN_DIR)/,$(REQUIRED_PROGRAMS))

demos: $(BIN_DIR) $(addprefix $(BIN_DIR)/,$(DEMO_PROGRAMS))

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(SRC_DIR)/%.c | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

run: all
	mpirun -np $(NP) $(BIN_DIR)/$(PROG) $(ARGS)

run-required: required
	@echo "Run examples:"
	@echo "  make run NP=4 PROG=q1_daxpy ARGS='65536 2.5'"
	@echo "  make run NP=4 PROG=q2_broadcast_race ARGS='10000000'"
	@echo "  make run NP=4 PROG=q3_distributed_dot ARGS='500000000 1.0'"
	@echo "  make run NP=4 PROG=q4_primes_master_slave ARGS='1000'"
	@echo "  make run NP=4 PROG=q5_perfect_master_slave ARGS='10000'"

run-demos: demos
	@echo "Run examples:"
	@echo "  make run NP=4 PROG=demo_blocking_ring ARGS='100'"
	@echo "  make run NP=4 PROG=demo_nonblocking_ring ARGS='10000000'"
	@echo "  make run NP=4 PROG=demo_pi_bcast_reduce ARGS='1000000'"

help:
	@echo "Build all programs: make"
	@echo "Build only required questions: make required"
	@echo "Build only demos: make demos"
	@echo "Run any program: make run NP=<p> PROG=<target> ARGS='<args>'"
	@echo "Available required targets: $(REQUIRED_PROGRAMS)"
	@echo "Available demo targets: $(DEMO_PROGRAMS)"

clean:
	rm -rf $(BIN_DIR)
