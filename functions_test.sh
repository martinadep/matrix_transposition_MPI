#!/bin/bash

# Define variables
SOURCE_FILES="tests/test_check_and_transp.c src/transp_functions.c src/checksym_functions.c src/utils.c"
OUTPUT_FILE="func_test.o"

# Create necessary directories
mkdir -p obj
echo "Setup complete."

# Compile the source files
module load mpich-3.2
mpicc -std=c11 -fopenmp $SOURCE_FILES -lm -o $OUTPUT_FILE
if [ $? -eq 0 ]; then
    echo "Compiled  successfully : $OUTPUT_FILE"
else
    echo "Error during compilation"
    exit 1
fi

echo "#############################################################"
# Loop to run the program with different processors number
for num_procs in 1 2 4 8 16 32 64 96; do
    mpiexec -np $num_procs ./$OUTPUT_FILE
    if [ $? -eq 0 ]; then
        echo "Executed successfully for num_procs: $num_procs"
        echo ""
    else
        echo "Error during execution for num_procs: $num_procs"
        echo ""
        continue
    fi
done
echo "#############################################################"
mv $OUTPUT_FILE obj