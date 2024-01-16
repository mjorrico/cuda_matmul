# Compiler and flags
NVCC := nvcc
GCC := g++
LDFLAGS := -lcudart -lcublas

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
KERNEL_DIR := src/kernels

# Source files
MAIN_SRC := main.cu
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
RUN_SRC := $(KERNEL_DIR)/run_kernel.cu
KERNEL_SRCS := $(wildcard $(KERNEL_DIR)/0*matmul*.cuh)

# Object files
MAIN_OBJ := $(OBJ_DIR)/main.o
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
RUN_OBJ := $(OBJ_DIR)/run_kernel.o

# Executable
TARGET := matmul

# Compilation
$(TARGET): $(MAIN_OBJ) $(CPP_OBJS) $(RUN_OBJ)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $(BIN_DIR)/$@ $(LDFLAGS)

$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $< -o $@ -I$(SRC_DIR) -I$(KERNEL_DIR)

$(RUN_OBJ): $(RUN_SRC) $(KERNEL_SRCS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $< -o $@ -I$(SRC_DIR) -I$(KERNEL_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(GCC) -c $< -o $@ -I$(SRC_DIR) -I$(KERNEL_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)