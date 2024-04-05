#!/bin/bash

# A Slurm batch file is, in short, a Bash script with some special comments that start with "#SBATCH".
# These special comments are used to specify command-line options for Slurm's "sbatch" command.
# For more info on what options are available and what they do, see the link below.
# https://slurm.schedmd.com/sbatch.html
# For more info on how to use these specifically for the Falcon HPC cluster, see the link below.
# https://sna.cs.colostate.edu/hpc/

# In general, Slurm will select which job to run next based on the following (in no particular order):
# 1. What resources are currently available
# 2. How many resources you requested
# 3. How long you want to use those resources for
# 4. How many jobs you've submitted recently
# 5. How many system resources you've used recently.
# Because of this, it's best to ask only for what you need.
# This will make it so your job is scheduled more quickly, prevent scheduling delays in the future,
# and it lets others use the cluster as well.

#######################################################################################################################
# Give your job a name so you can find it easier in the queue.
# Also indicate the name of the output file you'd like.
# If you don't specify an output file, Slurm will create one that looks like "slurm-1234.out",
# replacing "1234" with the job number.

#SBATCH --job-name="CSX75-Demo2"
#SBATCH --output="result.out"

#######################################################################################################################
# On the Falcon cluster, the partition selects what kind of nodes you want to use,
# and the QOS (quality of service) indicates how long you want to use it for.
# The "time" option lets us indicate the maximum time that our script will take (must be shorter than the QOS time).
# This is in "wall clock" time. I.e., if you ask for 5 minutes and have 5 tasks, the 5 minutes are NOT divided
# into 1 minute per task (which is different when compared to other resource requests, like using --mem for memory).

#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_debug
#SBATCH --time=00:05:00

#######################################################################################################################
# These specify how many resources we want to use.
# The descriptions here are how we will interpret them for now,
# but it's worth noting that they have other interpretations as well,
# and there are a variety of different ways to specify what you want.
# Some clusters may have limits on how many resources you can request at a time.
# For the Falcon resource limits, see the link below.
# https://sna.cs.colostate.edu/hpc/job-handling/limits/

# Nodes: the number of individual computers to use.
# GRES: "generic resources" to allocate per node. This requests two NVIDIA A100 40GB GPUs per node.
# GPU Bind: if and how to map GPUs to tasks. NCCL (communication library used by PyTorch) fails if you bind GPUs.
# Tasks Per GPU: the number of processes Slurm will start up per GPU.
# CPUs Per Task: the number of CPU cores used per task. This does not distinguish between threads and processes.
# Mem Per CPU: the amount of memory to allocate per CPU core (again, either threads or processes).

#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a100_3g.39gb:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M

#######################################################################################################################
# After specifying the options for the sbatch command, we tell Slurm what to do.
# Since we're using PyTorch, we need to use the module system to load the Anaconda instance.
# The final "srun" command indicates what task we want to be distributed.
# You can add some options to that as well to get a highly customized Slurm job, but we don't need that here.
# For information on what options are available, see the link below.
# https://slurm.schedmd.com/srun.html

# An important thing to note (in general, although not for this specific script):
# "srun" is able to detect if the program you're running supports MPI.
# If it does, then "srun" will work very similarly to "mpirun" or "mpiexec".

# For future scripts you may write, you are allowed to have multiple "srun" commands.
# These are called "job steps", and are used to have one job run multiple programs
# without needing to go through the scheduler twice.

# Final note: using "srun" is not strictly necessary, but it's highly recommended.
# See the links below for more information on why srun is useful and how it differs from sbatch.
# https://stackoverflow.com/questions/64327432/slurm-why-do-we-need-srun-in-sbatch-script-file
# https://stackoverflow.com/questions/43767866/slurm-srun-vs-sbatch-and-their-parameters

module purge
module load python/anaconda

srun python DistributedTraining.py
