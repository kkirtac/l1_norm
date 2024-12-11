import torch
from torch import nn
import pytorch_lightning as pl

class L1NormGRU(pl.LightningModule):
    """This RNN model updates the current hidden state and feeds it to the dense layer at each time step.
    The output of the dense layer accumulates the l1-norm prediction at current time step.
    """
    def __init__(self, hidden_size=64):
        """Initializes the network.

        Args:
            hidden_size (int, optional): Size of the hidden state vector. Defaults to 64.
        """
        super(L1NormGRU, self).__init__()
        self.hidden_size = hidden_size

        # GRU cell and fully connected layer
        self.rnn_cell = nn.GRUCell(input_size=1, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def compute_abs(self, x):
        """Compute the absolute value using allowed operations."""
        return self.relu(x) + self.relu(-x)

    def forward(self, x: torch.tensor, seq_lengths: list):
        """Forward pass to approximate the L1 norm.

        Args:
            x (torch.tensor): Input tensor of shape (batch_size, seq_length).
            seq_lengths (list): List indicating the original length of each sequence.

        Returns:
            torch.tensor: predicted l1 norm  of shape (batch_size,)
        """
        batch_size, seq_length = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)  # Initial hidden state

        # Mask to handle variable-length sequences
        mask = torch.arange(seq_length).unsqueeze(0) < torch.tensor(seq_lengths).unsqueeze(1)
        mask = mask.to(self.device).float()

        l1_norm_approx = torch.zeros(batch_size, device=x.device)

        for t in range(seq_length):
            inp = x[:, t].unsqueeze(1)  # (batch_size, 1)
            inp_abs = self.compute_abs(inp)  # Compute absolute value
            h = self.rnn_cell(inp_abs, h)  # Update GRU hidden state
            out = self.fc(h).squeeze(1)  # Compute contribution to L1 norm (batch_size,)
            
            # Only add contributions for valid timesteps based on the mask
            l1_norm_approx += out * mask[:, t]

        return l1_norm_approx

    def training_step(self, batch, batch_idx):
        """Performs one step of weight update with the given batch on the training set"""
        x, targets, seq_lengths = batch
        outputs = self(x, seq_lengths)
        # Mean Squared Error loss between the network output and the true L1 norm
        loss = nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
    
    def test_step(self, batch):
        """Performs one step of loss calculation with the given batch on the test set.
        The loss is accumulated for each batch and the average is reported at epoch end. 
        """
        x, targets, seq_lengths = batch
        outputs = self(x, seq_lengths)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    

class L1NormLSTM(pl.LightningModule):
    """This RNN model updates the current hidden state and feeds it to the dense layer at each time step.
    The output of the dense layer accumulates the l1-norm prediction at current time step.
    """
    def __init__(self, hidden_size=64):
        """Initializes the network.

        Args:
            hidden_size (int, optional): Size of the hidden state vector. Defaults to 64.
        """
        super(L1NormLSTM, self).__init__()
        self.hidden_size = hidden_size

        # GRU cell and fully connected layer
        self.rnn_cell = nn.LSTMCell(input_size=1, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def compute_abs(self, x):
        """Compute the absolute value using allowed operations."""
        return self.relu(x) + self.relu(-x)

    def forward(self, x: torch.tensor, seq_lengths: list):
        """Forward pass to approximate the L1 norm.

        Args:
            x (torch.tensor): Input tensor of shape (batch_size, seq_length).
            seq_lengths (list): List indicating the original length of each sequence.

        Returns:
            torch.tensor: predicted l1 norm  of shape (batch_size,)
        """
        batch_size, seq_length = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)  # Initial hidden state
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)  # Initial cell state

        # Mask to handle variable-length sequences
        mask = torch.arange(seq_length).unsqueeze(0) < torch.tensor(seq_lengths).unsqueeze(1)
        mask = mask.to(self.device).float()

        l1_norm_approx = torch.zeros(batch_size, device=x.device)

        for t in range(seq_length):
            inp = x[:, t].unsqueeze(1)  # (batch_size, 1)
            inp_abs = self.compute_abs(inp)  # Compute absolute value
            h, c = self.rnn_cell(inp_abs, (h,c))  # Update LSTM hidden state
            out = self.fc(h).squeeze(1)  # Compute contribution to L1 norm (batch_size,)
            
            # Only add contributions for valid timesteps based on the mask
            l1_norm_approx += out * mask[:, t]

        return l1_norm_approx

    def training_step(self, batch, batch_idx):
        """Performs one step of weight update with the given batch on the training set"""
        x, targets, seq_lengths = batch
        outputs = self(x, seq_lengths)
        # Mean Squared Error loss between the network output and the true L1 norm
        loss = nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
    
    def test_step(self, batch):
        """Performs one step of loss calculation with the given batch on the test set.
        The loss is accumulated for each batch and the average is reported at epoch end. 
        """
        x, targets, seq_lengths = batch
        outputs = self(x, seq_lengths)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
