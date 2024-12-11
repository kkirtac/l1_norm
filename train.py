"""Trains the RNN models. Abs function is computed by relu(x) + relu(-x).
The rnn learns to combine the abs outputs of the sequence to approximate the sum.
Outputs a csv file showing the results for different hyperparameter configurations.
"""
import pandas as pd
import pytorch_lightning as pl
from itertools import product
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset.random_sequence_dataset import RandomSequenceDataset, collate_fn
from model.rnn_model import L1NormGRU, L1NormLSTM


results = []

model_names = ["L1NormGRU", "L1NormLSTM"]
hidden_sizes= [16, 32, 64]
num_samples = [
    {"train":30000, "val":10000, "test":10000}, 
    {"train":80000, "val":30000, "test":30000}
]

max_length = [20, 40, 80]
input_range = [(-20, 20)]

# Compute the Cartesian product of hyperparameters
for num_sample, max_len, inp_range in product(num_samples, max_length, input_range):
    # Instantiate datasets and data loaders
    train_dataset = RandomSequenceDataset(num_samples=num_sample["train"], max_length=max_len, input_range=inp_range)
    val_dataset = RandomSequenceDataset(num_samples=num_sample["val"], max_length=max_len, input_range=inp_range)
    test_dataset = RandomSequenceDataset(num_samples=num_sample["test"], max_length=max_len, input_range=inp_range)

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=False, collate_fn=collate_fn)

    for model_name, hid_size in product(model_names, hidden_sizes):
        model_class = eval(model_name)
        model = model_class(hidden_size=hid_size)

        # Define the TensorBoard logger
        logger = TensorBoardLogger("tb_logs", name=f"my_model/{model_name}_{hid_size}_{num_sample['train']}_{max_len}_{inp_range[1]}")

        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

        # Train the model using PyTorch Lightning Trainer
        trainer = pl.Trainer(logger=logger, callbacks=[early_stop_callback, checkpoint_callback], max_epochs=50)
        trainer.fit(model, train_loader, val_loader)

        early_stop_epoch = None
        # Log the last epoch when training stopped
        if trainer.should_stop:
            early_stop_epoch = trainer.current_epoch

        test_loss = trainer.test(dataloaders=test_loader)

        test_error = test_loss[0]["test_loss"]

        results.append((model_name, hid_size, num_sample["train"], num_sample["val"], num_sample["test"], max_len, inp_range[0], inp_range[1], early_stop_epoch, test_error))

        df_result = pd.DataFrame(
            results, 
            columns=["model_name", "hidden_size", "train_count", "val_count", "test_count", "max_sequence_length", "input_min", "input_max", "early_stop_epoch", "test_error"]
        )
        df_result.to_csv("output/results_rnn.csv", index=False)
