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
        ## Hand-written cross entropy loss without speed-up tricks
        output_exp = torch.exp(output-torch.max(output, 1, keepdim=True)[0])
        output_exp_sum = output_exp.sum(1, keepdim=True)
        output_softmax = output_exp/output_exp_sum
        loss = -torch.log(output_softmax.gather(1, target.view(-1, 1))).mean()
        return loss
        ## CrossEntropy is also implemented in PyTorch with better numerical stability and speed
        # loss = nn.CrossEntropyLoss()(output, target)
        # return loss
    
class SimpleNCELoss(nn.Module):
    """
    In this SimpleNCELoss, we sample noise words from the vocabulary uniformly.
    It works well and fast for small vocabulary size.
    """
    def __init__(
        self,
        num_class: int,
        k: int = 1000,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(SimpleNCELoss, self).__init__()
        self.k = k
        self.num_class = num_class
        self.device = device
    
    def forward(self, output, target):
        # Sample noise words
        target = target.view(target.shape[0], target.shape[1], 1)
        noise = torch.randint(0, self.num_class, (target.shape[0], target.shape[1], self.k), device=self.device)
        # Calculate NCE loss
        target_scores = (output.gather(2, target))
        noise_scores = (output.gather(2, noise))
        all_scores = torch.cat((target_scores, noise_scores), 2)
        all_probs = torch.softmax(all_scores, 2)
        nce_loss = -torch.log(all_probs[:, :, 0]/(all_probs[:, :, 0] + torch.sum(all_probs[:, :, 1:], 2))).mean()
        return nce_loss
    
class NCELoss(nn.Module):
    def __init__(
        self,
        num_class: int,
        k: int = 1000,
        noise_distribution: torch.Tensor = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(NCELoss, self).__init__()
        self.k = k
        self.num_class = num_class
        if noise_distribution is None:
            self.noise_distribution = torch.random.uniform(0, 1, (num_class,)).to(device)
        else:
            self.noise_distribution = noise_distribution.view(-1).to(device)
        self.device = device
        
    def forward(self, output, target):
        batch_size = output.shape[0]
        max_length = output.shape[1]
        noise_distribution = self.noise_distribution.repeat(batch_size, max_length, 1).view(batch_size, max_length, -1)
        noise_distribution = noise_distribution.view(-1, noise_distribution.shape[-1])
        noise_distribution = noise_distribution / noise_distribution.sum(-1, keepdim=True)
        noise = torch.multinomial(noise_distribution.view(-1, noise_distribution.shape[-1]), self.k, replacement=False)
        output = output.view(-1, output.shape[-1])
        target = target.view(-1, 1)
        target_scores = output.gather(1, target)
        noise_scores = output.gather(1, noise)
        target_probn = noise_distribution.gather(1, target)
        noise_probn = noise_distribution.gather(1, noise)
        scores = torch.cat((target_scores, noise_scores), 1)
        probn = torch.cat((target_probn, noise_probn), 1)
        delta_s_theta = scores - torch.log(self.k * probn)
        nce_loss = -(torch.log(torch.sigmoid(delta_s_theta[:, 0])) + torch.sum(torch.log(1 - torch.sigmoid(delta_s_theta[:, 1:])), 1)).mean()
        return nce_loss
        
        

# class NCELoss(nn.Module):
#     def __init__(
#         self,
#         num_class: int,
#         k: int = 1000,
#         noise_distribution: np.ndarray = None,
#         device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     ):
#         super(NCELoss, self).__init__()
#         self.k = k
#         self.num_class = num_class
#         self.noise_distribution = torch.from_numpy(noise_distribution).reshape(-1).to(device) if noise_distribution is not None else None
#         self.device = device
    
#     def forward(self, output, target):
#         batch_size = output.shape[0]
#         max_length = output.shape[1]
#         if self.noise_distribution is None:
#             noise_distribution = torch.rand(batch_size, max_length, self.num_class, device=self.device)
#         else:
#             noise_distribution = self.noise_distribution.repeat(batch_size, max_length, 1)
#             noise_distribution = noise_distribution.view(batch_size, max_length, -1).to(self.device)
#         noise_distribution.scatter_(2, target.view(batch_size, max_length, 1), 0)
#         noise_distribution = noise_distribution.view(-1, noise_distribution.shape[-1])
#         noise_distribution = noise_distribution / noise_distribution.sum(-1, keepdim=True)
#         noise = torch.multinomial(noise_distribution.view(-1, noise_distribution.shape[-1]), self.k, replacement=False)
#         output = output.view(-1, output.shape[-1])
#         target = target.view(-1, 1)
#         target_scores = output.gather(1, target)
#         noise_scores = output.gather(1, noise)
#         all_scores = torch.cat((target_scores, noise_scores), 1)
#         all_probs = torch.softmax(all_scores, 1)
#         nce_loss = -torch.log(all_probs[:, 0]/(all_probs[:, 0] + self.k * torch.sum(all_probs[:, 1:], 1))).mean()
#         return nce_loss