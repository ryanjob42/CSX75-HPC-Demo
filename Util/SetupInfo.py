from dataclasses import dataclass
from torch.distributed import FileStore

import os
import torch.distributed as dist

@dataclass
class SetupInfo:
    """Holds information about how this script was distributed."""

    job_id: int
    """The ID number for the job which was run."""

    world_size: int
    """The number of instances of this script which were started."""
    
    rank: int
    """A 0-based ID number that uniquely identifies this process from all of the others in the same job."""

    local_rank: int
    """A 0-based ID number that uniquely identifies this process from the others on the same node,
    but NOT uniquely from processes on different nodes."""

    thread_count: int
    """A number of threads that this process is allowed to use."""
    
    def establish_communication(self) -> None:
        """Establishes communication with the other processes via PyTorch."""
        # We will use a file as a way to establish communication since all systems
        # in the CSU Computer Science department use a common networked file system.
        # It's important to give the file store a unique name, as if the file already exists,
        # the job will get stuck when trying to initialize the process group,
        # and you'll have to manually kill the job and start over.
        file_store = FileStore(f'filestore-{self.job_id}', self.world_size)

        # There are two main backends that PyTorch allows: 'nccl' and 'gloo'.
        # NCCL is the NVIDIA Collective Communications Library, and is designed for
        # inter-GPU communication. However, it is not currently compatible with
        # NVIDIA's Multi-Instance GPU (MIG) technology, which the Falcon cluster uses.
        # Because of this, we will use Gloo, which moves all tensors through the CPU, but always works.
        dist.init_process_group(backend='gloo', store=file_store, rank=self.rank, world_size=self.world_size)
        if not dist.is_initialized():
            raise ValueError('could not initialize the process group.')

    def is_main_process(self) -> bool:
        """Indicates if the current process is considered the 'main' one that leads the others."""
        return self.rank == 0
    
    def read_environment(self, variable: str) -> int:
        """Reads a system environment variable and interprets it as an integer."""
        return int(os.environ.get(variable))

class SlurmSetup(SetupInfo):
    """Collects the information about how this script was distributed via Slurm."""
    def __init__(self) -> None:
        super().__init__(
            job_id=self.read_environment('SLURM_JOB_ID'),
            world_size=self.read_environment('SLURM_NTASKS'),
            rank=self.read_environment('SLURM_PROCID'),
            local_rank=self.read_environment('SLURM_LOCALID'),
            thread_count=self.read_environment('SLURM_CPUS_PER_TASK')
        )
