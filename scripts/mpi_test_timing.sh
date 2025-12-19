#!/bin/bash
cd ..
# Define variables
SOURCE_FILES="./tests/test_timing.c src/transp_functions.c src/utils.c"
OUTPUT_FILE="mpi_test.o"
OUTPUTDIR="timing_out"

# Clean previous build and outputs
echo "Cleaning previous build and outputs..."
rm -rf obj $OUTPUTDIR *.o medie_mpi.csv

# Create necessary directories
echo "Setting up directories..."
mkdir -p $OUTPUTDIR
mkdir -p obj

echo "Setup complete."

# Compile the source files
#module load mpich-3.2
mpicc -std=c11 -fopenmp $SOURCE_FILES -lm -o $OUTPUT_FILE
if [ $? -eq 0 ]; then
    echo "Compiled  successfully : $OUTPUT_FILE"
else
    echo "Error during compilation"
    exit 1
fi
echo "matrix,time,procs,algo" >> medie_mpi.csv
echo "Please wait, it might take few minutes to complete all the executions"
echo "since the program is looped 10 times for each process number"
# Loop to run the program with different thread numbers
for num_procs in 1 2 4 8 16 32 64 96; do
    mpiexec -np $num_procs ./$OUTPUT_FILE $num_procs >> medie_mpi.csv

    if [ $? -eq 0 ]; then
        echo "Executed successfully for num_procs: $num_procs"
    else
        echo "Error during execution for num_procs: $num_procs"
        continue
    fi

done

mv $OUTPUT_FILE obj