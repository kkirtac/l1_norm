import torch
import numpy as np
from torch.utils.data import Dataset
    
class RandomSequenceDataset(Dataset):
    """Custom Dataset that generates random variable-length sequences and their L1 norms"""
    def __init__(self, num_samples, max_length, input_range=(-10, 10)):
        """Initializes the dataset.

        Args:
            num_samples (int): Number of sequences.
            max_length (int): Maximum sequence length. Defaults to 20.
            input_range (tuple, optional): Range of the input values. Defaults to (-10, 10).
        """
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_range = input_range
        np.random.seed(42) # seed for reproducibility

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Returns an input sequence and its l1 norm as output"""
        length = np.random.randint(1, self.max_length + 1)  # Random sequence length
        sequence = np.random.uniform(self.input_range[0], self.input_range[1], size=length).astype(np.float32)
        target = np.sum(np.abs(sequence)).astype(np.float32)  # L1 norm as the target
        return torch.tensor(sequence), torch.tensor(target)

def collate_fn(batch):
    """Custom collate function for padding variable-length sequences"""
    sequences, targets = zip(*batch)
    seq_lengths = [len(seq) for seq in sequences]
    max_length = max(seq.size(0) for seq in sequences)
    padded_sequences = torch.zeros(len(sequences), max_length)

    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.size(0)] = seq

    return padded_sequences, torch.tensor(targets), seq_lengths
