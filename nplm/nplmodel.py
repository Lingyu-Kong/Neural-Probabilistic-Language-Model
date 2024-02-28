import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nplm.data_loader.penn_loader import PennTreeBankLoader
from nplm.data_loader.vocab import Vocab, BOS, EOS
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
        weight_decay: float = 1e-5, 
        momentum: float = 0.9,
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
        optimizer = optimizer(self.logit_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.logit_model.train()
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            loss_list = []
            pbar = tqdm(data_loader, desc="Training Loss: ....")
            for num_batch, data_batch in enumerate(pbar):
                optimizer.zero_grad()
                batch_input, batch_target, _ = PennTreeBankLoader.process_batch(data_batch, sep_target=True)
                batch_input, batch_target = batch_input.to(self.device), batch_target.to(self.device)
                batch_output = self.logit_model(batch_input)
                loss = criterion(batch_output, batch_target)
                loss.backward()
                optimizer.step()
                ## `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
                torch.nn.utils.clip_grad_norm_(self.logit_model.parameters(), 0.25)
                optimizer.step()
                
                loss_list.append(loss.item())
                
                if (num_batch % log_interval == 0 and num_batch > 0) or (num_batch == len(data_loader) - 1):
                    cur_loss = np.mean(loss_list)
                    pbar.set_description(f"Training Loss: {cur_loss:.2f}")
        
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
        
                
                

from nplm.models.lstm.lstm import LSTMModel
from nplm.data_loader.penn_loader import PennTreeBank, PennTreeBankLoader
from nplm.criterion.train_criterion import CrossEntropyLoss, SimpleNCELoss
from nplm.criterion.eval_criterion import PPL, WER
  
if __name__ == "__main__":
    vocab = Vocab(
        files=["./data_loader/data/penn_train.txt"],
    )
    print("Length of the Vocabulary: ", len(vocab))
    train_dataset = PennTreeBank(
        file_path="./data_loader/data/penn_train.txt",
        vocab=vocab,
    )
    train_dataloader = PennTreeBankLoader(
        dataset=train_dataset,
        vocab=vocab,
        batch_size=256,
        shuffle=True,
    )
    
    lstm = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    )
    nplmodel = NPLModel(
        logit_model=lstm,
        vocab=vocab,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    # nplmodel.train(
    #     data_loader=train_dataloader,
    #     optimizer=torch.optim.SGD,
    #     criterion=CrossEntropyLoss(),
    #     epochs=50,
    #     lr=1.0,
    #     weight_decay=1e-5,
    #     momentum=0.9,
    # )
    
    nplmodel.train(
        data_loader=train_dataloader,
        optimizer=torch.optim.SGD,
        criterion=SimpleNCELoss(k=1000, vocab=vocab),
        epochs=50,
        lr=1.0,
        weight_decay=1e-5,
        momentum=0.9,
    )
    
    test_dataset = PennTreeBank(
        file_path="./data_loader/data/penn_test.txt",
        vocab=vocab,
    )
    test_dataloader = PennTreeBankLoader(
        dataset=test_dataset,
        vocab=vocab,
        batch_size=256,
        shuffle=True,
    )
    ppl_loss = nplmodel.evaluate(
        data_loader=test_dataloader,
        criterion=PPL(),
    )
    print(f"Perplexity Loss: {ppl_loss:.2f}")
    wer_loss = nplmodel.evaluate(
        data_loader=test_dataloader,
        criterion=WER(),
    )
    print(f"Word Error Rate Loss: {wer_loss:.2f}")
    
    assert vocab.query("we") != -1
    output_string = nplmodel.generate(start_token="we", max_length=100)
    print(output_string)
    
    os.makedirs("./save", exist_ok=True)
    nplmodel.save(path="./save")
    nplmodel = NPLModel.load(path="./save")
    assert vocab.query("he") != -1
    output_string = nplmodel.generate(start_token="he", max_length=100)
    print(output_string)