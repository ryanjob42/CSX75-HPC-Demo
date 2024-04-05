# These first two lines allow us to import classes from Util.
import sys
sys.path.append('../Util')

from SetupInfo import SlurmSetup

from dataclasses import dataclass
from torch import Tensor
from torch.nn import Linear, Module, NLLLoss, Sequential
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import numpy
import os
import torch
import torch.distributed as distributed

BATCH_SIZE = 40
EPOCH_COUNT = 3
TRAIN_DATA_SIZE = 4000
TEST_DATA_SIZE = 80

class QuadraticData(Dataset):
    """A dataset representing quadratic equations.
    Each equation will be of the form "ax^2 + bx + c".
    The neural network will take a, b, c, and x as inputs.
    """
    
    def __init__(self, size: int) -> None:
        # For the quadratic equations, we need four values per input: a, b, c, and x.
        # We'll construct this as a 2D tensor.
        # The DataLoader expects the first dimension to be a single sample
        # and all other dimensions to be a single input into the model.
        # For our purpose, we only need two dimensions: one of length "size",
        # and the other of length 4 (for a, b, c, and x respectively).
        self.inputs = torch.rand((size, 4))

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        """Returns two elements: the input to your model and the expected output of the model.
        Both values must be tensors."""
        input_value = self.inputs[idx]
        return input_value, self.compute(input_value)
    
    def compute(self, input: Tensor) -> Tensor:
        """Performs the quadratic computation desired."""
        (a,b,c,x) = input
        return (a*x*x) + (b*x) + c

class MyNeuralNetowrk(Module):
    """The neural network model to train."""

    def __init__(self):
        super().__init__()

        # We'll simply make a 4-layer neural network.
        # See the link below for different kinds of neural network layers PyTorch has.
        # https://pytorch.org/docs/stable/nn.html
        self.layers = Sequential(
            Linear(4, 10),      # 4 inputs, 10 outputs
            Linear(10, 20),     # 10 inputs, 20 outputs
            Linear(20, 5),      # 20 inputs, 5 outputs
            Linear(5, 1)        # 5 inputs, 1 output
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

def main() -> None:
    """The entry point for this script."""
    
    # Get the information about how this script was distributed.
    # PyTorch doesn't automatically establish communication, so we have to do it ourselves.
    setup = SlurmSetup()
    print(f'Rank {setup.rank}: starting up.')
    setup.establish_communication(use_gpus=True)
    print(f'Rank {setup.rank}: communication is ready.')

    # All processes create a data loader from our custom quadratic dataset and a distributed sampler.
    # We'll force PyTorch to use a specific seed so all instances of the script generate the same data.
    torch.manual_seed(0)
    dataset = QuadraticData(TRAIN_DATA_SIZE)
    sampler = DistributedSampler(dataset, num_replicas=setup.world_size, rank=setup.rank)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    # All processes create an instance of our model and wrap it in Distributed Data Parallel,
    # which will handle all the distributed communication for us.
    # We'll put it on the GPU with the same ID as this process's local rank (the rank on this node).
    # Note: setting find_unused_parameters=True is required for the backward pass (i.e., training).
    # See the link below under the "Forward Pass" bullet for more information.
    # https://pytorch.org/docs/stable/notes/ddp.html#internal-design
    base_model = MyNeuralNetowrk().to(setup.local_rank)
    model = DistributedDataParallel(base_model, find_unused_parameters=True)

    # For training, we will need a loss function and an optimizer.
    # See the links below for different kinds of loss functions and optimizers PyTorch has.
    # https://pytorch.org/docs/stable/nn.html
    # https://pytorch.org/docs/stable/optim.html
    loss_function = NLLLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model for the desired number of epochs.
    for epoch in range(EPOCH_COUNT):
        print(f'Rank {setup.rank}: starting training epoch {epoch}.')

        # Put the model into training mode.
        model.train()

        # Iterate through all batches in the data loader.
        for batch_index, (input_batch, target_batch) in enumerate(data_loader):
            # Send the data to the GPU.
            input_batch = input_batch.to(setup.local_rank)
            target_batch = target_batch.to(setup.local_rank)

            # Clear the gradient from the previous batch and do a forward pass through the model.
            # PyTorch automatically handles the fact that we've put an entire batch of data into the model at once.
            optimizer.zero_grad()
            output_batch = model(input_batch)

            # Compute the loss and use that to do a backwards pass through our model.
            # Then take a step of our optimizer.
            # Since we're using DDP, PyTorch handles all the communications for us.
            loss_batch = loss_function(output_batch, target_batch)
            loss_batch.backward()
            optimizer.step()

            print(f'Rank {setup.rank}: completed training batch {batch_index}.')
        
        print(f'Rank {setup.rank}: completed epoch {epoch}.')
        
        # Save a copy of the model at the end of each epoch.
        # Only the main process should do this, otherwise all the processes
        # will write to the same file at the same time, corrupting it.
        if setup.is_main_process():
            torch.save(model.state_dict(), f'model_epoch_{epoch}')

    # Evaluate the model using a brand new dataset.
    # We'll only do this on the main node, that way we don't have to combine several loss outputs.
    if setup.is_main_process():
        # We'll still put the testing data into a data loader, but we don't need a distributed sampler.
        dataset = QuadraticData(TRAIN_DATA_SIZE)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

        # Put the model into evaluation mode, tell PyTorch we don't need to compute gradients right now,
        # then run through all the data to compute our loss.
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for input_batch, target_batch in data_loader:
                # Move our intput and target to the GPU.
                input_batch = input_batch.to(setup.local_rank)
                target_batch = target_batch.to(setup.local_rank)
                
                output_batch = model(input_batch)

                # Calling ".item()" moves the value from the GPU to the CPU.
                test_loss += loss_function(output_batch, input_batch).item()

        print(f'Rank {setup.rank}: The final loss is: {test_loss}.')
    
    print(f'Rank {setup.rank}: Done training.')

if __name__ == '__main__':
    main()
