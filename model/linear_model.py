import torch
from torch import nn
import pytorch_lightning as pl

class L1NormLinear(pl.LightningModule):
    """Model that approximates the abs function to calculate the l1 norm"""
    def __init__(self, hidden_size=32):
        """Initializes the network.
        The network calculates per element outputs to approximate the abs function,
        and then sums all outputs to predict the l1 norm.

        Args:
            hidden_size (int, optional): Size of the latent vector. Defaults to 32.
        """     
        super(L1NormLinear, self).__init__()
        self.hidden_size = hidden_size
        # Per-element network to approximate the absolute value function
        self.per_element_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.tensor, seq_lengths: list):
        """Forwards one batch to predict the l1 norm.
        One batch contains multiple sequences of same length (after padding).

        Args:
            x (torch.tensor): a batch of inputs, with shape (batch_size, max_seq_len)
            seq_lengths (list): length of each sequence of the batch before padding.

        Returns:
            torch.tensor: predicted l1 norm
        """
        # x: [batch_size, max_seq_len]
        batch_size, max_seq_len = x.size()
        # Reshape x to [batch_size * max_seq_len, 1]
        x = x.view(-1, 1)
        # Apply per-element network
        y = self.per_element_net(x)
        # Reshape y back to [batch_size, max_seq_len]
        y = y.view(batch_size, max_seq_len)
        # Create mask to zero out padded elements
        mask = torch.arange(max_seq_len).unsqueeze(0) < torch.tensor(seq_lengths).unsqueeze(1)
        mask = mask.to(self.device).float()
        y = y * mask
        # Sum over sequence dimension to get the L1 norm approximation
        output = y.sum(dim=1)
        return output

    def training_step(self, batch: torch.tensor, batch_idx):
        """Performs one step of weight update with the given batch on the training set"""
        x, targets, seq_lengths = batch
        outputs = self(x, seq_lengths)
        # Mean Squared Error loss between the network output and the true L1 norm
        loss = nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.tensor, batch_idx):
        """Performs one step of loss calculation with the given batch on the validation set.
        The loss is accumulated for each batch and the average is reported at epoch end. 
        """
        x, targets, seq_lengths = batch
        outputs = self(x, seq_lengths)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Using Adam optimizer
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def test_step(self, batch: torch.tensor):
        """Performs one step of loss calculation with the given batch on the test set.
        The loss is accumulated for each batch and the average is reported at epoch end. 
        """
        x, targets, seq_lengths = batch
        outputs = self(x, seq_lengths)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    