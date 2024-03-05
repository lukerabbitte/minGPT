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
        user_id = 4
        eval_states, eval_actions, eval_rewards, eval_timesteps = self.eval_dataset[user_id]
        reward_sum = 0

        x = torch.tensor([4]).unsqueeze(0).unsqueeze(-1)
        y = torch.tensor([116]).unsqueeze(0).unsqueeze(-1)
        print(f"original y shape: {y.shape}")
        r = torch.tensor([5]).unsqueeze(0).unsqueeze(-1)
        t = torch.tensor([76]).unsqueeze(0).unsqueeze(-1)

        for i in range(10):
            print(f"x..shape: {x.shape}")
            print(f"y.shape: {y.shape}")
            print(f"r.shape: {r.shape}")
            print(f"t.shape: {t.shape}")
            sampled_action = sample(self.model, x, 1, temperature=10.0, sample=True, top_k=None, actions=y, rewards=r, timesteps=t)
            action = sampled_action.squeeze(0).squeeze(0)
            state_tensor = torch.tensor([[[4]]])
            x = torch.cat((x, state_tensor), dim=1)
            action_tensor = torch.tensor([[[action]]])
            print(f"action tensor size: {action_tensor.shape}")
            y = torch.cat((y, action_tensor), dim=1)
            action = action + 1
            print(f"sampled_action was: {action} for user: {4}")
            action_index = np.where(eval_actions == action)
            reward = eval_rewards[action_index]
            print(f"reward for this action was: {reward} for user: {4}")
            reward_sum += reward
            reward_tensor = torch.tensor([[[reward]]])
            r = torch.cat((r, reward_tensor), dim=1)
            timestep_tensor = torch.tensor([[[i + 1]]])
            t = torch.cat((t, timestep_tensor), dim=1)

        print("Desired return across 10-recommendation sequence for user 4: %d, Actual average return: %d" % (50, reward_sum))
        self.model.train(True)
        return reward_sum
