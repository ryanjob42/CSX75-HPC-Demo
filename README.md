# CS475/575 HPC Demo (Lab 8)
A demonstration on how to use the Falcon HPC cluster at Colorado State University (CSU). This was built for CS475/575 Parallel Programming/Processing Lab 8.

## CSU Falcon Cluster
The Falcon cluster[^falcon] is a high-performance compute (HPC) cluster available through the Computer Science department. This cluster uses a common job scheduling and management system called Slurm[^slurm]. This acts as a way to request system resources for you to use. It then schedules jobs fairly, allowing many people to use the cluster (both simultaneously and over time).

## Demo 1: Distributed Sum
The first demonstration here is for PyTorch's distributed communications library[^dist], which contains MPI-like functions for you to use. Our main node will generate a set of random values and scatter them across the cluster. Each node will then compute the sum of its subset, and these partial sums will be gathered. The main node finishes the summation and print the result.

## Demo 2: Distributed Neural Network Training
The second demonstration here is for PyTorch's approach to using a distributed model. This focuses on the Distributed Data Parallel (DDP) class[^ddp], which duplicates your model across all tasks (i.e., instances of your script), then evenly divides the work to do. To help with the division of work, we use the Dataset and DataLoader classes[^dataset]. These classes use the MPI-like calls from Demo 1 under the hood so you don't have to. However, understanding what's going on behind the scences is still important so you can get good performance!

This demo simply trains a very basic neural network on a quadratic equation for a few epochs, then checks how far off the model is.

## Running the Demos
You can run both of these demos in the same way.

1. SSH into the Falcon cluster.
   1. `ssh user@falcon.cs.colostate.edu`
   2. Make sure you're either on campus or using the VPN!
2. Clone this repository.
   1. `git clone git@github.com:ryanjob42/CSX75-HPC-Demo.git`
3. Navigate into the Demo 1 folder using the `cd` command.
   1. For demo 1: `cd CSX75-HPC-DEMO/Demo1`
   2. For demo 2: `cd CSX75-HPC-DEMO/Demo2`
4. Launch the demo using the `sbatch` command.
   1. `sbatch start-falcon.slurm`
   2. This command will print your job ID number to the console.
5. Once launched, you can use the `squeue` command to check if your job is waiting, running, or complete.
   1. To see your jobs specifically, you can use `squeue --me`.
   2. If your job is complete, it won't show up.
6. After the job completes, the outputs will be stored in the file `result.out`.

## Using Other Clusters
The Falcon cluster has pre-installed all the Python packages you will need for this demo. However, if you use a different cluster, this may not be the case. Luckily, these demos only require two packages: NumPy and PyTorch. You can find instructions on how to install these packages from their websites[^numpy] [^pytorch]. I have also created a [Conda environment file](./environment.yml) that you can use. Conda's website has some excellent instructions on how to manage Conda environments[^conda1], including how to create an environment from a file[^conda2]. If the cluster you are using does not have Python installed at all, the easiest solution is to use Miniconda[^miniconda], which is free and does not require admin privilege to install.

<!-- References -->
[^falcon]: CSU Falcon Cluster: https://sna.cs.colostate.edu/hpc/
[^slurm]: Slurm Workload Manager: https://slurm.schedmd.com/overview.html
[^dist]: PyTorch Distributed Communications Library: https://pytorch.org/docs/stable/distributed.html
[^ddp]: PyTorch Distributed Data Parallel (DDP) Overview: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
[^dataset]: PyTorch Datasets and DataLoaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
[^numpy]: NumPy Website: https://numpy.org/
[^pytorch]: PyTorch Website: https://pytorch.org/
[^conda1]: Managing Conda Environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
[^conda2]: Create a Conda Environment from a File: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
[^minicondad]: Minicona Website: https://docs.anaconda.com/free/miniconda/index.html
