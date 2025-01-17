#!/bin/bash
cd ..
# Define variables
SOURCE_FILES="src/main.c src/checksym_functions.c src/transp_functions.c src/utils.c"
OUTPUT_FILE="midtermII.o"

# Change this to run with different matrix size
MATRIX_POW="12"


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
echo "Running with [$((2**MATRIX_POW))]x[$((2**MATRIX_POW))] matrix"
echo ""
# Loop to run the program with different processors number
for num_procs in 1 2 4 8 16 32 64 96; do
    mpiexec -np $num_procs ./$OUTPUT_FILE $MATRIX_POW

    if [ $? -eq 0 ]; then
        echo "Executed successfully for num_procs: $num_procs"
        echo ""
    else
        echo "Error during execution for num_procs: $num_procs"
        echo ""
        continue
    fi

done

echo ""
echo "#############################################################"
mv $OUTPUT_FILE obj