## MPI matrix transposition
Explicit parallelization of matrix transpose operation 
using Message Passing Interface (MPI).

This project aims to benchmark and analyze the performance of this approach,
comparing its efficiency and scalability with OpenMP and implicit parallelization 
approaches explored in [this project](https://github.com/martinadep/matrix_transposition_OpenMP).

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
    - [On HPC Cluster](#on-hpc-cluster)
    - [On your local machine](#on-your-local-machine)
---

# Project Structure

```bash
├── scripts
│   ├── functions_test.sh
│   ├── mpi_test_timing.sh
│   ├── run_main.sh
├── src
│   ├── checksym_functions.c
│   ├── main.c
│   ├── main.h
│   ├── transp_functions.c
│   ├── utils.c
├── tests
│   ├── test_check_and_transp.c
│   ├── test_timing.c
│   ├── test_timing.h
├── functions_test.pbs
├── mpi_test_timing.pbs
├── .gitignore
├── README.md
├── run_main.pbs
```
---

# How to Reproduce

## On HPC Cluster

### Prerequisites
- Access to the HPC cluster (in this case of the University of Trento)
- SSH client

### Steps
1. Open an ssh session on University of Trento's HPC cluster, in your preferred terminal
```bash
$ ssh <username>@hpc.unitn.it
```
2. Clone the repository
```bash
$ git clone https://github.com/martinadep/parco_second
```
3. Navigate to the folder
```bash
$ cd parco_second/
```

**!!! remember to change home directory path in each .pbs before submitting !!!**

4. Run the following command:
```bash
$ qsub run_main.pbs
```
## Additional tests
This section shows how to run additional tests:
- (1) testing two MPI approaches for matrix transposition with different number of processes
    - Using MPI_Bcast()
    - Using MPI_Datatype() and MPI_Scatterv()
```bash
$ qsub mpi_test_timing.pbs
```
- (2) assess correctness of matrix transposition and check symmetry functions

```bash
$ qsub functions_test.pbs
```

## On your local machine

### Prerequisites
- mpich compiler

### Steps
1. Clone the repository
```bash
$ git clone https://github.com/martinadep/parco_second
```
2. Navigate the repository folder
```bash
$ cd parco_second/src
```
3. Run the following commands:
```bash
$ mpicc -std=c11 main.c checksym_functions.c transp_functions.c utils.c -lm -fopenmp -o matrix_transposition.o
$ ./matrix_transposition.o <select_matrix_size>
```

## Additional tests
This section shows how to run additional tests
1. Testing two MPI approaches for matrix transposition with different number of processes
    - Using MPI_Bcast()
    - Using MPI_Datatype() and MPI_Scatterv()
```bash
$ cd parco_second/scripts/
$ ./mpi_test_timing.sh
```
2. Assess correctness of matrix transposition and check symmetry functions

```bash
$ cd parco_second/scripts/
$ ./functions_test.sh
```

## Data Analysis
You can process the data and plot the graphs using the provided python script in the `analysis/` folder.
