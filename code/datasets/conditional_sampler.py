import random
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
# https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

class ConditionalBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, protected_attribute):
        protected_attribute = protected_attribute.int()
        assert len(torch.unique(protected_attribute)) == 2 # assert binary attribute
        self.first_group_indices = [i for i in range(len(protected_attribute)) if protected_attribute[i] == 0]
        self.second_group_indices = [i for i in range(len(protected_attribute)) if protected_attribute[i] == 1]
        self.batch_size = batch_size
    
    def __iter__(self):
        random.shuffle(self.first_group_indices)
        random.shuffle(self.second_group_indices)
        first_group_batches  = chunk(self.first_group_indices, self.batch_size)
        second_group_batches = chunk(self.second_group_indices, self.batch_size)
        combined = list(first_group_batches + second_group_batches)
        combined = [batch.tolist() for batch in combined]
        random.shuffle(combined)
        return iter(combined)
    
    def __len__(self):
        return (len(self.first_group_indices) + len(self.second_group_indices)) // self.batch_size

