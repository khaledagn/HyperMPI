#!/bin/bash

# test script for HyperMPI

echo "Starting integration test..."

# test if MPI is installed and running
if ! mpirun --version &> /dev/null; then
    echo "Error: MPI is not installed or configured properly."
    exit 1
fi

# check if Python is installed
if ! python3 --version &> /dev/null; then
    echo "Error: Python is not installed."
    exit 1
fi

# create temporary hostfile
echo "Creating temporary hostfile..."
TEMP_HOSTFILE=$(mktemp)
echo -e "127.0.0.1 slots=4" > $TEMP_HOSTFILE
echo "Hostfile created at: $TEMP_HOSTFILE"

# create temporary Python script for testing
echo "Creating temporary Python script..."
TEMP_PYTHON_SCRIPT=$(mktemp --suffix=.py)
cat <<EOL > $TEMP_PYTHON_SCRIPT
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from process {rank} out of {size} processes!")
EOL
echo "Python script created at: $TEMP_PYTHON_SCRIPT"

# run a sample distributed training script using mpirun
echo "Running the distributed training script..."
mpirun --allow-run-as-root -n 4 --hostfile $TEMP_HOSTFILE python3 $TEMP_PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "Error: Distributed training script failed."
    exit 1
else
    echo "Integration test passed! Distributed training script ran successfully."
fi

# clean up temporary files
echo "Cleaning up..."
rm -f $TEMP_HOSTFILE
rm -f $TEMP_PYTHON_SCRIPT

echo "Test completed."
