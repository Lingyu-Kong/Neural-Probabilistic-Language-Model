import torch
import torch.nn as nn
import numpy as np

## Criterion class in eval_criterion.py is not garanteed to be differentiable

class PPL(nn.Module):
    def __init__(self):
        super(PPL, self).__init__()

    def forward(self, output, target):
        target = target.view(-1)
        output = output.view(-1, output.size(-1))
        loss = nn.CrossEntropyLoss(reduction='none')(output, target)
        loss = loss.view(target.size(0), -1).mean(dim=-1)
        loss = loss.mean()
        return torch.exp(loss)

class WER(nn.Module):
    def __init__(self):
        super(WER, self).__init__()

    def forward(self, output, target):
        output_probs = torch.softmax(output, dim=-1)
        output_words = torch.argmax(output_probs, dim=-1).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        wer = self.edit_distance(output_words, target)
        
        wer = torch.mean(torch.from_numpy(wer.astype(float)) / target.shape[1])
        
        return wer

    def edit_distance(self, output_words: np.ndarray, target: np.ndarray) -> np.ndarray:
        dp = np.zeros((output_words.shape[0], output_words.shape[1] + 1, target.shape[1] + 1)).astype(np.float32)
        
        for i in range(output_words.shape[1] + 1):
            dp[:, i, 0] = i
        for j in range(target.shape[1] + 1):
            dp[:, 0, j] = j

        for i in range(1, output_words.shape[1] + 1):
            for j in range(1, target.shape[1] + 1):
                mask = np.array((output_words[:, i - 1] == target[:, j - 1]))
                mask = mask.astype(np.float32)
                dp[:, i, j] = mask * dp[:, i - 1, j - 1] + (1 - mask) * (np.minimum(np.minimum(dp[:, i - 1, j], dp[:, i, j - 1]), dp[:, i - 1, j - 1]) + 1)

        return dp[:, -1, -1]