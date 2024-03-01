import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nplm.data_loader.penn_loader import PennTreeBankLoader
from nplm.models.lstm.lstm import LSTMModel
from nplm.data_loader.vocab import Vocab, BOS, EOS
from nplm.criterion.eval_criterion import PPL, WER
import numpy as np
from tqdm import tqdm
import os

class NPLModel(object):
    """
    Neural Probabilistic Language Model.
    Args:
        logit_model (nn.Module): Logit model, e.g. LSTM. The ouputs are passed through a softmax layer. 
        vocab (Vocab): Vocabulary for the language model.
        device (torch.device): Device to run the model.
    """
    def __init__(
        self, 
        logit_model: nn.Module,
        vocab: Vocab,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.logit_model = logit_model
        self.vocab = vocab
        self.device = device
        self.logit_model.to(self.device)
        
    def train(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 40,
        lr: float = 1.0,
        lr_scheduler: bool = False,
        val_data_loader: DataLoader = None,
        log_interval: int = 100,
    ):
        """
        Train the model with the given data loader.
        Args:
            data_loader (DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
            criterion (nn.Module): Loss function.
            epochs (int): Number of epochs to train the model.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            momentum (float): Momentum for the optimizer.
            log_interval (int): Log interval for the training.
        """
        optimizer = optimizer(self.logit_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        self.logit_model.train()
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            loss_list = []
            pbar = tqdm(data_loader, desc="Training Loss: ...., Validation PPL: ...., Validation WER: ....")
            for num_batch, data_batch in enumerate(pbar):
                optimizer.zero_grad()
                batch_input, batch_target, _ = PennTreeBankLoader.process_batch(data_batch, sep_target=True)
                batch_input, batch_target = batch_input.to(self.device), batch_target.to(self.device)
                batch_output = self.logit_model(batch_input)
                loss = criterion(batch_output, batch_target)
                loss.backward()
                ## `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
                torch.nn.utils.clip_grad_norm_(self.logit_model.parameters(), 0.25)
                optimizer.step()
                
                loss_list.append(loss.item())
                
                if (num_batch % log_interval == 0 and num_batch > 0) or (num_batch == len(data_loader) - 1):
                    if num_batch == len(data_loader) - 1:
                        cur_loss = np.mean(loss_list)
                        ppl = self.evaluate(val_data_loader, PPL())
                        wer = self.evaluate(val_data_loader, WER())
                        pbar.set_description(f"Training Loss: {cur_loss:.2f}, Validation PPL: {ppl:.2f}, Validation WER: {wer:.2f}")
                    else:
                        cur_loss = np.mean(loss_list)
                        pbar.set_description(f"Training Loss: {cur_loss:.2f}, Validation PPL: ...., Validation WER: ....")
            if lr_scheduler:             
                scheduler.step(ppl)
        
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
    ):
        """
        Evaluate the model with the given data loader.
        Args:
            data_loader (DataLoader): DataLoader for the evaluation data.
            criterion (nn.Module): Loss function.
        """
        self.logit_model.eval()
        loss_list = []
        with torch.no_grad():
            for num_batch, data_batch in enumerate(data_loader):
                batch_input, batch_target, effective_length = PennTreeBankLoader.process_batch(data_batch, sep_target=True)
                batch_input, batch_target = batch_input.to(self.device), batch_target.to(self.device)
                batch_output = self.logit_model(batch_input)
                loss = criterion(batch_output, batch_target)
                loss_list.append(loss.item())
        self.logit_model.train()
        return np.mean(loss_list)
    
    def generate(
        self,
        start_token: str,
        max_length: int = 100,
    ):
        """
        Generate a sequence from the model.
        Args:
            start_token (str): Start token for the sequence.
            max_length (int): Maximum length of the sequence.
        """
        input = torch.LongTensor([self.vocab.word2index[BOS], self.vocab.word2index[start_token]]).to(self.device)
        input = input.unsqueeze(0)
        for i in range(max_length):
            output = self.logit_model(input)
            output = torch.softmax(output, dim=-1)[0, -1, :]
            max_idx = torch.argmax(output)
            if max_idx == self.vocab.word2index[EOS]:
                break
            output = max_idx.unsqueeze(0).unsqueeze(0)
            input = torch.cat((input, output), dim=1)
        output_words = []
        for i in range(1, input.size(1)):
            output_words.append(self.vocab.index2word[input[0, i].item()])
        output_string = " ".join(output_words)
        return output_string
    
    def save(self, path: str):
        """
        Save the model.
        Args:
            path (str): Directory path to save the model.
        """
        self.logit_model.save(os.path.join(path, "model.pth"))
        self.vocab.save(os.path.join(path, "vocab.pkl"))

    @staticmethod
    def load(path: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> 'NPLModel':
        """
        Load the model.
        Args:
            path (str): Directory path to load the model.
        """
        logit_model = LSTMModel.load(os.path.join(path, "model.pth"))
        vocab = Vocab.load(os.path.join(path, "vocab.pkl"))
        return NPLModel(logit_model=logit_model, vocab=vocab, device=device)