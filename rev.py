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
from mingpt.rev_utils import read_data
import argparse
from mingpt.rev_utils import plot_loss

seed = 123
epochs = 10
batch_size = 64
context_length = 30
model_type = 'reward_conditioned'
# num_steps = 500000
# block_size = 30 * 3

class ReviewDataset(Dataset):

    def __init__(self, states, actions, rewards, returns, returns_to_go, timesteps, terminal_indices, block_size):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.returns = returns
        self.returns_to_go = returns_to_go
        self.timesteps = timesteps
        self.terminal_indices = terminal_indices
        self.block_size = block_size
        self.vocab_size = max(actions) + 1

    def __len__(self):
        return len(self.states) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3   # aka, the original context length
        done_idx = idx + block_size
        for i in self.terminal_indices:
            if i > idx:  # find the first terminal index greater than idx
                done_idx = min(i, done_idx)
                break

        idx = done_idx - block_size

        # Squeeze these tensors to give dimension for batch size expected by most APIs (b,t)
        # Notice that original paper didn't unsqueeze these until later
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # was (block_size, 1) back when there was an unsqueeze
        returns_to_go = torch.tensor(self.returns_to_go[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)
        # print(f"states.size: {states.shape}")
        # print(f"actions.size: {actions.shape}")
        # print(f"rewards.size: {rewards.shape}")
        # print(f"timesteps.size: {timesteps.shape}")

        return states, actions, returns_to_go, timesteps


class EvalDataset(Dataset):

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
        start_indices = start_indices + self.terminal_indices[:-1]

        for i in self.terminal_indices:
            if i > idx:  # find the first terminal index greater than idx
                done_idx = i
                break

        lookup = self.terminal_indices.index(done_idx)
        if len(start_indices) == len(self.terminal_indices):
            start_idx = start_indices[lookup]

        # Squeeze these tensors to give dimension for batch size expected by most APIs (b,t)
        # Notice that original paper didn't unsqueeze these until later
        states = torch.tensor(np.array(self.states[start_idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[start_idx:done_idx], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(self.rewards[start_idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        # print(f"states.shape: {states.shape}")
        # print(f"timesteps.shape: {timesteps.shape}")
        # print(f"idx: {idx} and timesteps: {timesteps}")
        return states, actions, rewards, timesteps

# Read in train data and create dataset
train_states, train_actions, train_rewards, train_returns, train_returns_to_go, train_timesteps, train_terminal_indices = read_data(
    'data/goodreads_eval_modified_20pc.tsv')
train_dataset = ReviewDataset(train_states, train_actions, train_rewards, train_returns, train_returns_to_go, train_timesteps, train_terminal_indices, context_length * 3)
len_train_dataset = len(train_states)

# Read in test data and create dataset
# test_states, test_actions, test_rewards, test_timesteps, test_terminal_indices = read_data('test_dummy_50.tsv')
# test_dataset = ReviewDataset(test_states, test_actions, test_rewards, test_timesteps, test_terminal_indices, context_length * 3)
# len_test_dataset = len(test_states)

eval_states, eval_actions, eval_rewards, _, _, eval_timesteps, eval_terminal_indices = read_data(
    'data/goodreads_eval_modified.tsv')
eval_dataset = EvalDataset(eval_states, eval_actions, eval_rewards, eval_timesteps, eval_terminal_indices, context_length * 3)
len_eval_dataset = len(eval_states)

# print(f"max_timesteps across entire dataset is: {max(timesteps)}")
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8,
                  n_embd=128, model_type=model_type, max_timestep=max(train_timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size, learning_rate=0.0048,
                      lr_decay=False, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * context_length * 3,
                      num_workers=4, seed=seed, model_type=model_type,
                      ckpt_path="checkpoints/model_checkpoint.pth",
                      max_timestep=max(train_timesteps))
trainer = Trainer(model, train_dataset, None, tconf, eval_dataset)
train_losses, test_losses, average_rewards_per_epoch = trainer.train()

plot_loss(train_losses, None, context_length, batch_size,
          mconf.n_layer, mconf.n_head, mconf.n_embd, 'data/goodreads_eval_modified_20pc.tsv', len_train_dataset, None, None, tconf.learning_rate, tconf.lr_decay)

print(f"train_losses: {train_losses}")
print(f"average_rewards_per_epoch: {average_rewards_per_epoch}")
# print(f"test_losses: {test_losses}")
