import torch
import torch.nn as nn
import os

class RNNModel(nn.Module):
    """
    RNN Model for language modeling. Base Model for the NPLM.
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the word embeddings.
        hidden_dim (int): Dimension of the hidden state.
        num_layers (int): Number of RNN layers.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
    
    def save(self, path: str):
        """
        Save the model.
        Args:
            path (str): File path to save the model.
        """
        model_params = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            'state_dict': self.state_dict(),
        }
        
        torch.save(model_params, path)
        
    @staticmethod
    def load(path: str) -> 'RNNModel':
        """
        Load the model.
        Args:
            path (str): File path to load the model.
        """
        model_params = torch.load(path)

        model = RNNModel(
            vocab_size=model_params['vocab_size'],
            embedding_dim=model_params['embedding_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            dropout=model_params['dropout'],
        )
        model.load_state_dict(model_params['state_dict'])
        
        return model
