import os
import json
import random
import torch

# Get total number of parameters in a model
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_json(fp):
    with open(fp) as json_file:
        data = json.load(json_file)
    return data

def rindex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def pad(seqs, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    return padded.to(device), lens

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)
