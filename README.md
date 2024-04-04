# CSX75-HPC-Demo
A demonstration on how to use the Falcon HPC cluster at CSU. This was built for CS475/575 Parallel Programming/Processing Lab 6.

## Demo 1: Distributed Sum
The first demonstration is for PyTorch's distributed communications library[^1], which contains MPI-like functions for you to use. Our main node will generate a set of random values and scatter them across the cluster. Each node will then compute the sum of its subset, and these partial sums will be gathered. The main node finishes the summation and print the result.

## References
[^1]: PyTorch Distributed Communications Library: https://pytorch.org/docs/stable/distributed.html
