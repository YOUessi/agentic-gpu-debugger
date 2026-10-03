PYTHON ?= python3
NVCC ?= nvcc
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/case0022
KERNEL := benchmarks/public/case_0022/public_input/kernel.cu
HARNESS := benchmarks/harness/vector_io.cpp

.PHONY: check compile clean

check:
	$(PYTHON) scripts/check_case0022.py

compile:
	mkdir -p $(BUILD_DIR)
	$(NVCC) -std=c++17 -lineinfo \
		-Ibenchmarks/harness -Ibenchmarks/harness/vendor \
		$(KERNEL) $(HARNESS) -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
