# Demo 1: Distributed Sum
This demonstration uses PyTorch's distributed communications library[^dist], which contains MPI-like functions for you to use.

At a high level, here is what this demo does:

1. The processes all establish communication with each other.
2. The "main" process (rank 0) generates some random numbers.
3. The random numbers are "scattered" so each process gets a subset of them.
4. Each process computes the sum of its subset.
5. All the partial sums are "gathered" to the main process.
6. The main process computes a final sum.

<!-- References -->
[^dist]: PyTorch Distributed Communications Library: https://pytorch.org/docs/stable/distributed.html
