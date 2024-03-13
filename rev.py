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
from mingpt.rev_utils import plot_reward

seed = 123
epochs = 30
batch_size = 64
context_length = 30
model_type = 'reward_conditioned'
train_dataset_filename = 'data/goodreads_eval_80pc_constant_state.tsv'
eval_dataset_filename = 'data/goodreads_eval.tsv'
eval_data_filename = 'data/goodreads_eval_first_50.tsv'

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

        # Get indices denoting start of each user's interaction trajectory
        self.start_indices = self.terminal_indices
        self.start_indices = np.insert(self.start_indices, 0, 0)

    def __len__(self):
        return len(self.states)

    # Returns user data from a complete matrix of user interactions where they have rated every item, for eval purposes
    # Also note that each of our terminal indices will be one index higher than the last entry for a user.
    # This is in keeping with Python's upper bound exclusive behaviour.
    def __getitem__(self, user_id):
        print(f"user_id passed to EvalDataset getitem is: {user_id}")
        idx = self.start_indices[user_id - 1]
        done_idx = None if user_id == self.start_indices.size else self.terminal_indices[
            user_id - 1]  # avoid array out of limit bug for last user

        # Return tensors of (episode_length, 1)
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rewards, timesteps

# Read in train data and create dataset
train_states, train_actions, train_rewards, train_returns, train_returns_to_go, train_timesteps, train_terminal_indices = read_data(
    train_dataset_filename)
train_dataset = ReviewDataset(train_states, train_actions, train_rewards, train_returns, train_returns_to_go, train_timesteps, train_terminal_indices, context_length * 3)
len_train_dataset = len(train_states)

# Read in test data and create dataset
# test_states, test_actions, test_rewards, test_timesteps, test_terminal_indices = read_data('test_dummy_50.tsv')
# test_dataset = ReviewDataset(test_states, test_actions, test_rewards, test_timesteps, test_terminal_indices, context_length * 3)
# len_test_dataset = len(test_states)

eval_states, eval_actions, eval_rewards, _, _, eval_timesteps, eval_terminal_indices = read_data(
    eval_dataset_filename)
eval_dataset = EvalDataset(eval_states, eval_actions, eval_rewards, eval_timesteps, eval_terminal_indices, context_length * 3)
len_eval_dataset = len(eval_states)

# Forget above, do this
eval_data = pd.read_csv(eval_data_filename, sep='\t')

# print(f"max_timesteps across entire dataset is: {max(timesteps)}")
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8,
                  n_embd=128, model_type=model_type, max_timestep=max(train_timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size, learning_rate=0.0048,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * context_length * 3,
                      num_workers=4, seed=seed, model_type=model_type,
                      ckpt_dir="checkpoints/41",
                      max_timestep=max(train_timesteps),
                      num_users=256,
                      ratings_per_user=55,
                      num_recs=1,
                      ratings_at_extreme=False)
trainer = Trainer(model, train_dataset, None, tconf, eval_dataset, eval_data)
train_losses, action_losses, test_losses, rewards_per_epoch = trainer.train()

plot_loss(train_losses, None, context_length, batch_size,
          mconf.n_layer, mconf.n_head, mconf.n_embd, train_dataset_filename, len_train_dataset, None, None, tconf.learning_rate, tconf.lr_decay, tconf.num_users, tconf.ratings_at_extreme)

plot_reward(rewards_per_epoch, context_length, batch_size, mconf.n_layer, mconf.n_head, mconf.n_embd,
              train_dataset_filename, len_train_dataset, tconf.learning_rate, tconf.lr_decay, tconf.num_users, tconf.num_recs, tconf.ratings_at_extreme)

print(f"train_losses: {train_losses}")
print(f"rewards_per_epoch: {rewards_per_epoch}")
print(context_length, batch_size, mconf.n_layer, mconf.n_head, mconf.n_embd,
              train_dataset_filename, len_train_dataset, tconf.learning_rate, tconf.lr_decay, tconf.num_users, tconf.num_recs)
