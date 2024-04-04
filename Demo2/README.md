# Demo 2: Distributed Neural Network Training
This demonstration uses PyTorch's Distributed Data Parallel (DDP) class[^ddp] to distribute the training of a simple neural network model across multiple GPUs. It also uses PyTorch's Dataset and DataLoader classes[^dataset] to help ensure efficient data loading. This is paired with the Distributed Sampler class[^sampler] which ensures that the data is split properly for distributed training.

In short, here is how these classes are supposed to work together:
1. A Dataset simply holds the data that you want to use during training (or testing).
2. The Distributed Sampler is used to ensure each instance of the script only accesses a distinct subset of the data.
3. A Data Loader is given a Dataset (and a Distributed Sampler in this case) and handles pulling data from the Dataset prior to using it, then freeing the memory when you're done with it.

## Handling Large Amounts of Data
This demo only uses a small amount of data, so we create an inefficient Dataset that simply stores everything at all times. This isn't a viable solution if you're training a model on a large amount of data (e.g., image data). In this case, you'll want to use one of PyTorch's Data Pipe classes[^datapipe].

Data Pipes inherit from the Dataset class, allowing you to pass it in to the Data Loader. There are a wide variety of them to handle different kinds of data, but the common theme is that they efficiently load data as needed and free memory when the data is done.

[^ddp]: PyTorch Distributed Data Parallel (DDP) Overview: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
[^dataset]: PyTorch Datasets and DataLoaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
[^sampler]: PyTorch Distributed Sampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
[^datapipe]: PyTorch DataPipes: https://pytorch.org/data/main/dp_tutorial.html
