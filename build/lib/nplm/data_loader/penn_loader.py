import os
import torch
from torch.utils.data import Dataset, DataLoader
from nplm.data_loader.vocab import Vocab, UNK, BOS, EOS
from tqdm import tqdm

class PennTreeBank(Dataset):
    """
    Torch Dataset for the Penn Tree Bank dataset.
    Args:
        file_path (str): Path to the PTB dataset file.
        max_length (int): Maximum length of the sequences, items longer than this will be truncated.
        vocab: Vocabulary object.
    """
    def __init__(self, file_path: str, max_length: int, vocab: Vocab):
        self.file_path = file_path
        self.max_length = max_length
        self.vocab = vocab
        self.load_data(file_path)
        
    def load_data(self, file_path: str):
        """
        Load the data from the file.
        Args:
            file_path (str): Path to the PTB dataset file.
        """
        assert os.path.exists(file_path), f"File not found at {file_path}"
        with open(file_path, "r") as f:
            sentences = []
            for sentence in tqdm(f, desc="Loading data in {}".format(file_path)):
                sentences.append(sentence.split())
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        sentence = [self.vocab.word2index.get(word, self.vocab.word2index["<unk>"]) for word in sentence]
        if len(sentence) > self.max_length:
            sentence = sentence[:self.max_length]
        return torch.tensor(sentence, dtype=torch.long)q
        