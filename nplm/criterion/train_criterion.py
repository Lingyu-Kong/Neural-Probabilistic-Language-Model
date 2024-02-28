import torch
import torch.nn as nn
import numpy as np
from nplm.data_loader.vocab import Vocab, BOS, EOS

## Criterion classes in train_criterion.py should be differentiable and can be used as loss functions in the training process.

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        target = target.view(-1)
        output = output.view(-1, output.size(-1))
        return nn.CrossEntropyLoss()(output, target)

class SimpleNCELoss(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        k: int = 1000,
    ):
        super(SimpleNCELoss, self).__init__()
        self.k = k
        self.vocab = vocab
    
    def forward(self, output, target):
        # Sample noise words
        target = target.view(target.shape[0], target.shape[1], 1)
        noise = torch.randint(0, len(self.vocab), (target.shape[0], target.shape[1], self.k), device=output.device)
        # Calculate NCE loss
        target_scores = (output.gather(2, target))
        noise_scores = (output.gather(2, noise))
        all_scores = torch.cat((target_scores, noise_scores), 2)
        all_probs = torch.softmax(all_scores, 2)
        nce_loss = -torch.log(all_probs[:, :, 0]/(all_probs[:, :, 0] + torch.sum(all_probs[:, :, 1:], 2))).mean()
        return nce_loss
