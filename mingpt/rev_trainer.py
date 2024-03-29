"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from mingpt.rev_utils import sample
import random


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, eval_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.eval_dataset = eval_dataset
        self.train_losses = []
        self.test_losses = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            # print(f"is_train: {is_train}")
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            # Note that x,y,r are of size torch.Size([30]), or context length, and t of size ([1])
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # print(f"Train?: {is_train} - what does x of size: {x.size()} look like?: x[1]")  # ([128, 30, 1])
                # print(f"Train?: {is_train} - what does y of size: {y.size()} look like?: y[1]")  # ([128, 30, 1])
                # print(f"Train?: {is_train} - what does r of size: {r.size()} look like?: r[1]")  # ([128, 30, 1])
                # print(f"Train?: {is_train} - what does t of size: {t.size()} look like?: t[1]")  # ([128, 1, 1])

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

                    # print(f"x[-1].shape is: {x[-1].shape}")  # ([30, 1])

                    # lets sample
                    # sampled_action = sample(model, x[-1].unsqueeze(0), 1, temperature=10.0, sample=True, top_k=None, actions=y[-1].unsqueeze(0), rewards=r[-1].unsqueeze(0), timesteps=t[-1].unsqueeze(0))
                    # print(f"sampled_action was: {sampled_action.squeeze(0).squeeze(0) + 1} for user: {x[-1][1]}")
                    # print(sampled_action.shape)

                    self.get_returns()

            if is_train:
                self.train_losses.append(float(np.mean(losses)))
                print(f"train_loss is: {float(np.mean(losses))}")

            if not is_train:
                test_loss = float(np.mean(losses))
                self.test_losses.append(test_loss)
                print(f"test_loss is: {float(np.mean(losses))}")
                logger.info("test loss: %f", test_loss)
                return test_loss



        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')

            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            #  if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()


        return self.train_losses, self.test_losses

    def get_returns(self):

        self.model.train(False)

        # Get 10 unique users
        user_ids = random.sample(range(1, 256 + 1), 10)

        # Will contain all the total rewards per user episode
        total_rewards = []

        for user_id in user_ids:

            # Get a complete matrix for each user showing their interaction history
            eval_states, eval_actions, eval_rewards, eval_timesteps = self.eval_dataset[user_id]
            # rtgs = [ret]
            reward_sum = 0
            sampled_actions = []
            sampled_rewards = []

            for i in range(10):
                # State is simply userID at the moment, so we can start at any arbitrary point
                state = eval_states[i]
                print(f"state in get_returns looks like: {state}") # tensor([1.])
                state = state.unsqueeze(0).unsqueeze(-1).to(self.device)
                all_states = state if i == 0 else torch.cat([all_states, state], dim=1)

                print(f"all_states.shape: {all_states.shape}")

                # Handle initial case where state is just one state and actions are none
                # rewards needs to be None at beginning, but once we use rtg it has initial value.
                sampled_action = sample(self.model, all_states, 1, temperature=1.0, sample=True,
                                        actions=None if i == 0 else torch.tensor(all_actions, dtype=torch.long).to(
                                            self.device),
                                        rewards=None if i == 0 else torch.tensor(sampled_rewards, dtype=torch.long).to(self.device).unsqueeze(
                                            0).unsqueeze(-1),
                                        timesteps=(min(i, self.config.max_timestep) * torch.ones((1, 1, 1),
                                                    dtype=torch.int64).to(self.device)))

                print(f"sampled_action shape: {sampled_action.shape}") # IF 2 DIM THEN UNSQUEEZE
                # Find the reward corresponding to the generated action from our eval dataset
                action = sampled_action.squeeze(0).squeeze(0)
                action += 1  # action straight from model is 0-indexed, we want 1-indexed
                print(f"Action for user {user_id} was {action}")
                all_actions = torch.cat([all_actions, sampled_action], dim=1) # IF 2 DIM THEN WON'T WORK

                action_index = np.where(eval_actions == action)[0][0]
                print(f"This action corresponds to line: {action_index}")
                reward = eval_rewards[action_index] # rewards_user is a simple numpy array so no reshaping needed
                # print(f"Reward for user {user_id} was {reward}")

                sampled_actions += [sampled_action]
                sampled_rewards += [reward]

            total_rewards.append(reward_sum)
            # print(f"Recommended 10 new items to the user of id \"{user_id}\", wanted an accumulative total of {ret}, "
            #       f"got {reward_sum}")

        eval_return = sum(total_rewards) / 10.
        print("Desired average return across ten 10-recommendation sequences: %d, Actual average return: %d" % (50, eval_return))
        self.model.train(True)
        return eval_return