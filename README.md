# CSX75-HPC-Demo
A demonstration on how to use the Falcon HPC cluster at Colorado State University (CSU). This was built for CS475/575 Parallel Programming/Processing Lab 6.

## CSU Falcon Cluster
The Falcon cluster[^falcon] is a high-performance compute (HPC) cluster available through the Computer Science department. This cluster uses a common job scheduling and management system called Slurm[^slurm]. This acts as a way to request system resources for you to use. It then schedules jobs fairly, allowing many people to use the cluster (both simultaneously and over time).

## Demo 1: Distributed Sum
The first demonstration here is for PyTorch's distributed communications library[^dist], which contains MPI-like functions for you to use. Our main node will generate a set of random values and scatter them across the cluster. Each node will then compute the sum of its subset, and these partial sums will be gathered. The main node finishes the summation and print the result.

To run this demo:
1. SSH into the Falcon cluster.
   1. `ssh user@falcon.cs.colostate.edu`
   2. Make sure you're either on campus or using the VPN!
2. Clone this repository.
   1. `git clone git@github.com:ryanjob42/CSX75-HPC-Demo.git`
3. Navigate into the Demo 1 folder.
   1. `cd CSX75-HPC-DEMO/Demo1`
4. Launch the demo using the `sbatch` command.
   1. `sbatch falcon-start.slurm`
   2. This command will print your job ID number to the console.
5. Once launched, you can use the `squeue` command to check if your job is waiting, running, or complete.
   1. To see your jobs specifically, you can use `squeue --me`.
   2. If your job is complete, it won't show up.
6. After the job completes, the outputs will be stored in the file `result.out`.

## References
[^falcon]: CSU Falcon Cluster: https://sna.cs.colostate.edu/hpc/
[^dist]: PyTorch Distributed Communications Library: https://pytorch.org/docs/stable/distributed.html
[^slurm]: Slurm Workload Manager: https://slurm.schedmd.com/overview.html
