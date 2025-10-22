# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -pedantic -fopenmpgit

# Directories
SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Executable
TARGET = $(BIN_DIR)/les_solver

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories if missing
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean build
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

# Run
run: $(TARGET)
	./$(TARGET)
