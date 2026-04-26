NVCC = nvcc
CFLAGS = -O2 -arch=sm_86
LDFLAGS = -lm -lcublas -lcudnn

all: ex01 ex02 ex03 ex04 ex05

ex01: ex01_cuda_basics.cu
	$(NVCC) $(CFLAGS) $< -o $@

ex02: ex02_memory_hierarchy.cu
	$(NVCC) $(CFLAGS) $< -o $@

ex03: ex03_ml_primitives.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lm

ex04: ex04_cnn_layers.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcublas

ex05: ex05_mnist_cnn.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcudnn -lcublas

clean:
	rm -f ex01 ex02 ex03 ex04 ex05
