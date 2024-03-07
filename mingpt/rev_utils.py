import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rewards=None, timesteps=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:]
        rewards = rewards if rewards.size(1) <= block_size//3 else rewards[:, -block_size//3:]
        logits, _ = model(x_cond, actions=actions, targets=None, rewards=rewards, timesteps=timesteps)

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # logits = logits / temperature
        # print(f"logits shape is: {logits.shape}")

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # print(f"probs looks like: {probs.shape}")
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix
        # print(f"x.shape: {x.shape}")

    return x


def get_terminal_indices(arr):
    idxs = {}
    for i in reversed(range(len(arr))):
        idxs[arr[i]] = i
    done_idxs = list(idxs.values())
    done_idxs.reverse()
    done_idxs = done_idxs[1:]
    done_idxs.append(len(arr))
    return done_idxs


def get_next_filename(figs_dir, base_filename):
    i = 1
    while True:
        new_filename = f"{base_filename}_{i}.svg"
        if not os.path.exists(os.path.join(figs_dir, new_filename)):
            return new_filename
        i += 1


def plot_loss(train_losses, test_losses, context_length, batch_size, n_layer, n_head, n_embd,
              filename_train_dataset, len_train_dataset, filename_test_dataset, len_test_dataset, learning_rate, lr_decay, figs_dir='figs'):
    plt.rcParams.update({'font.family': 'monospace'})
    plt.figure(figsize=(12, 8))

    # Plot training loss
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')

    if test_losses is not None:
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='orange')

    plt.xlabel('Epoch', fontweight='bold', fontsize=14)
    plt.ylabel('Loss', fontweight='bold', fontsize=14)
    plt.title('Training and Test Loss Over Epochs', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Information box
    info_text = f"Context Length: {context_length}\nBatch Size: {batch_size}\nLayers: {n_layer}\nHeads: {n_head}\nEmbedding Size: {n_embd}\nTrain Dataset Name: {filename_train_dataset}\nTrain Dataset Size: {len_train_dataset}\nTest Dataset Name: {filename_test_dataset}\nTest Dataset Size: {len_test_dataset}\nLearning Rate: {learning_rate}\nLearning Rate Decay: {lr_decay}"
    plt.text(0.02, 0.85, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    if figs_dir:
        os.makedirs(figs_dir, exist_ok=True)
        base_filename = 'loss_plot_with_info'
        new_filename = get_next_filename(figs_dir, base_filename)
        plt.savefig(os.path.join(figs_dir, new_filename), format='svg')


def read_data(file_path):
    data = pd.read_csv(file_path, delimiter="\t")
    states = data['user_id'].tolist()
    actions = data['item_id'].tolist()
    actions = [a - 1 for a in actions]
    rewards = data['rating'].tolist()
    returns = []
    returns_to_go = np.zeros_like(rewards)
    timesteps = data['timestep'].tolist()
    terminal_indices = get_terminal_indices(states)  # Assuming get_terminal_indices is defined elsewhere

    start_index = 0
    # Generate returns-to-go
    for i in terminal_indices:
        rewards_by_episode = rewards[start_index:i]
        # print(f"rewards_by_episode: {rewards_by_episode} for data up to done_idx: {i}\n")
        returns = np.append(returns, sum(rewards_by_episode))
        for j in range(i - 1, start_index - 1, -1):
            rewards_by_reverse_growing_episode = rewards_by_episode[j - start_index:i - start_index]
            # print(f"rewards_by_reverse_growing_episode: {rewards_by_reverse_growing_episode}\n")
            returns_to_go[j] = sum(rewards_by_reverse_growing_episode)
            # print(f"returns_to_go: {returns_to_go}")
        start_index = i

    print(f"returns: {len(returns)}\n")
    print(f"returns_to_go: {returns_to_go}")

    return states, actions, rewards, returns, returns_to_go, timesteps, terminal_indices