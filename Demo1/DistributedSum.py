# These first two lines allow us to import classes from Util.
import sys
sys.path.append('../Util')

from SetupInfo import SlurmSetup

import numpy as np
import torch.distributed as dist

def main() -> None:
    """The entry point for this script."""
    
    # Get the information about how this script was distributed.
    # PyTorch doesn't automatically establish communication, so we have to do it ourselves.
    setup = SlurmSetup()
    print(f'Rank {setup.rank}: starting up.')
    setup.establish_communication(use_gpus=False)
    print(f'Rank {setup.rank}: communication is ready.')

    # For this demo, we will simply have each node compute the sum of a bunch of random numbers.
    # The main process will generate these numbers, then scatter them to everyone else.
    if setup.is_main_process():
        # Create a 2D array with one row per node and many values per row.
        source_data = np.random.rand(setup.world_size, 1000)
    else:
        # To make Python and PyTorch happy, the other processes need to have the same
        # variable, but set to "None" instead of the random data.
        source_data = None
    
    # To scatter the data, we need somewhere to store the subset belonging to this process.
    # PyTorch requires that this be a list containing "None", and will throw exceptions
    # if you don't do it this way (this is a common theme for these functions).
    my_subset = [None]
    dist.scatter_object_list(my_subset, source_data)

    # Compute the sum of the given subset and print it to the console.
    partial_sum = np.sum(my_subset)
    print(f'Rank {setup.rank}: my partial sum is {partial_sum}.')

    # To gather the partial sums, the main process needs somewhere to store them all.
    # PyTorch requies that this be a list with a length equal to the world size,
    # where each element is just "None". The other processes don't need anything,
    # but to make Python happy, we create the same variable and set it to "None".
    if setup.is_main_process():
        gathered_sums = [None] * setup.world_size
    else:
        gathered_sums = None
    dist.gather_object(partial_sum, object_gather_list=gathered_sums)

    # Finally, the main thread computes and prints the final sum.
    if setup.is_main_process():
        final_sum = np.sum(gathered_sums)
        print(f'Rank {setup.rank}: the final sum is {final_sum}.')

# When running distributed Python code, it's best practice to check if
# this script is being run directly, then call a "main" method if so.
# If you don't, you may accidentally run it many more times than you intended,
# especially if you're using some of PyTorch's other distributed processing libraries.
if __name__ == '__main__':
    main()
