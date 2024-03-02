import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from mingpt.utils import set_seed
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from mingpt.rev_model import GPT, GPTConfig
from mingpt.rev_trainer import Trainer, TrainerConfig
from mingpt.rev_utils import get_terminal_indices
import argparse
from mingpt.rev_utils import plot_loss

seed = 123
epochs = 10
batch_size = 128
context_length = 30
# num_steps = 500000
# block_size = 30 * 3

class ReviewDataset(Dataset):

    def __init__(self, states, actions, rewards, timesteps, terminal_indices, block_size):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.timesteps = timesteps
        self.terminal_indices = terminal_indices
        self.block_size = block_size
        self.vocab_size = max(actions) + 1

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        start_indices = [0]
        start_indices = start_indices + terminal_indices[:-1]

        for i in self.terminal_indices:
            if i > idx:  # find the first terminal index greater than idx
                done_idx = i
                break

        lookup = terminal_indices.index(done_idx)
        if len(start_indices) == len(terminal_indices):
            start_idx = start_indices[lookup]

        # Squeeze these tensors to give dimension for batch size expected by most APIs (b,t)
        # Notice that original paper didn't unsqueeze these until later
        states = torch.tensor(np.array(self.states[start_idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[start_idx:done_idx], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(self.rewards[start_idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        print(f"states.shape: {states.shape}")
        print(f"idx: {idx} and timesteps: {timesteps}")
        return states, actions, rewards, timesteps

# Read in data
data = pd.read_csv('goodreads_sparse.tsv', delimiter="\t")
states = data['user_id'].tolist()
# print(f"states: {states}")
actions = data['item_id'].tolist()
actions = [a - 1 for a in actions]
rewards = data['rating'].tolist()
timesteps = data['timestep'].tolist()
terminal_indices = get_terminal_indices(states)

# Train
train_dataset = ReviewDataset(states, actions, rewards, timesteps, terminal_indices, context_length * 3)
len_train_dataset = len(states)
# print(f"max_timesteps across entire dataset is: {max(timesteps)}")
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8,
                  n_embd=128, max_timestep=max(timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size, learning_rate=0.0048,
                      lr_decay=False, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * context_length * 3,
                      num_workers=4, seed=seed,
                      ckpt_path="checkpoints/model_checkpoint.pth",
                      max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, None, tconf)
train_losses = trainer.train()

# plot_loss(train_losses, None, context_length, batch_size,
#           mconf.n_layer, mconf.n_head, mconf.n_embd, 'dummy_50.tsv', len_train_dataset, None, None, tconf.learning_rate, tconf.lr_decay)

print(f"train_losses: {train_losses}")
