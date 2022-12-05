import torch

def save_checkpoint(state, is_best, best_filename):
    if is_best:
        torch.save(state, best_filename)