# Compiler
CC := gcc
# Compiler flags
CFLAGS := -Wall -Wextra -I mfcc_inc -I nnom_inc

# Source directories
SRC_DIRS := mfcc_src nnom_src
# Include directories
INC_DIRS := mfcc_inc nnom_inc

# Source files
SRC_FILES := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c))
# Object files
OBJ_FILES := $(patsubst %.c,%.o,$(SRC_FILES))

EXEC := my_program

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ_FILES) main.o
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ_FILES) main.o $(EXEC)