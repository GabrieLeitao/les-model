# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -pedantic -fopenmp

# Directories
SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin

# --- CONFIGURATION ---

# 1. List the .cpp file for your MAIN simulation (the one with main())
#    (Fixed the typo from .cwhpp to .cpp)
MAIN_SRC = $(SRC_DIR)/main.cpp

# 2. List the .cpp file for your TEST (the one you just wrote with main())
TEST_SRC = $(SRC_DIR)/test_checkpoint.cpp

# 3. List ALL OTHER .cpp files that are SHARED between them
SHARED_SRCS = $(SRC_DIR)/pressure.cpp \
              $(SRC_DIR)/io.cpp \
              $(SRC_DIR)/statistics.cpp

# --- END CONFIGURATION ---

# Define Executables
TARGET = $(BIN_DIR)/les_solver
TEST_TARGET = $(BIN_DIR)/test_checkpoint

# --- Automatic Object File Generation ---
MAIN_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(MAIN_SRC))
TEST_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TEST_SRC))
SHARED_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SHARED_SRCS))

# Default target
all: $(TARGET)

# Test target: build the test executable, then run it
test: $(TEST_TARGET)
	@echo "--- Running Checkpoint Test ---"
	./$(TEST_TARGET)
	@echo "--- Test Finished ---"

# Build main executable
$(TARGET): $(MAIN_OBJ) $(SHARED_OBJS) | $(BIN_DIR)
	@echo "Linking Main Solver: $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build test executable
$(TEST_TARGET): $(TEST_OBJ) $(SHARED_OBJS) | $(BIN_DIR)
	@echo "Linking Test Executable: $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

# Generic rule to compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories if missing
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean build
clean:
	@echo "Cleaning build files..."
	rm -rf $(OBJ_DIR)/*.o $(TARGET) $(TEST_TARGET)

# Run main simulation
run: $(TARGET)
	./$(TARGET)

.PHONY: all test clean run
