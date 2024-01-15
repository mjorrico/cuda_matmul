# Compiler and flags
NVCC := nvcc
GCC := g++
LDFLAGS := -lcudart

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
KERNEL_DIR := src/kernels

# Source files
MAIN_SRC := main.cu
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
KERNEL_SRCS := $(wildcard $(KERNEL_DIR)/*cu)

# Object files
MAIN_OBJ := $(OBJ_DIR)/main.o
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
KERNEL_OBJS := $(patsubst $(KERNEL_DIR)/%.cu, $(OBJ_DIR)/%.o, $(KERNEL_SRCS))

# Executable
TARGET := matmul

# Compilation
$(TARGET): $(MAIN_OBJ) $(CPP_OBJS) $(KERNEL_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $(BIN_DIR)/$@ $(LDFLAGS)

$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $< -o $@ -I$(SRC_DIR) -I$(KERNEL_DIR)

$(OBJ_DIR)/%.o: $(KERNEL_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $< -o $@ -I$(SRC_DIR) -I$(KERNEL_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(GCC) -c $< -o $@ -I$(SRC_DIR) -I$(KERNEL_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)