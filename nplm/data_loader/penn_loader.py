import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
    def __init__(self, file_path: str, vocab: Vocab, max_length: int = None):
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
                sentences.append([BOS] + sentence.split() + [EOS])
        self.data = sentences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx]
        if self.max_length is not None:
            sentence = sentence[:self.max_length]
        sentence = [self.vocab.word2index.get(word, self.vocab.word2index[UNK]) for word in sentence]
        return sentence
    
class ConcatPennTreeBank(Dataset):
    """
    Torch Dataset for the Penn Tree Bank dataset.
    Args:
        file_path (str): Path to the PTB dataset file.
        max_length (int): Maximum length of the sequences, items longer than this will be truncated.
        vocab: Vocabulary object.
    """
    def __init__(self, file_path: str, vocab: Vocab, max_length: int = 30):
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
                sentences += [BOS] + sentence.split() + [EOS]
        self.data = sentences
    
    def __len__(self):
        return len(self.data) // self.max_length
    
    def __getitem__(self, idx):
        sentence = self.data[idx*self.max_length:(idx+1)*self.max_length]
        sentence = [self.vocab.word2index.get(word, self.vocab.word2index[UNK]) for word in sentence]
        return sentence
    

def pad_collate_fn(batch):
    """Pad the list of word indexes into 2-D LongTensor"""
    length = [len(sentence) for sentence in batch]
    return pad_sequence([torch.LongTensor(s) for s in batch], batch_first=True), torch.LongTensor(length)
    
    
class PennTreeBankLoader(DataLoader):
    """
    Torch DataLoader for the Penn Tree Bank dataset.
    Args:
        file_path (str): Path to the PTB dataset file.
        max_length (int): Maximum length of the sequences, items longer than this will be truncated.
        vocab: Vocabulary object.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers.
    """
    def __init__(self, dataset: Dataset, vocab: Vocab, batch_size: int = 16, shuffle: bool = False, num_workers: int = 1):
        self.dataset = dataset
        self.vocab = vocab
        super(PennTreeBankLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
        )
        
    @staticmethod
    def process_batch(batch, sep_target=True):
        """
        Process the padded batch.
        Args:
            batch (list): List of sentences.
            sep_target (bool): Whether to separate the target from the input.
        """
        batch_sentences, batch_lengths = batch
        max_length = max(batch_lengths)
        batch_sentences = batch_sentences[:, :max_length]
        effective_length = max_length - 1 ## -1 for the EOS token. Excluding the EOS token for the loss calculation.
        if sep_target:
            batch_input = batch_sentences[:, :-1].contiguous()
            batch_target = batch_sentences[:, 1:].contiguous()
            return batch_input, batch_target, effective_length
        else:
            batch_input = batch_sentences.contiguous()
            batch_target = batch_sentences.contiguous()
            return batch_input, batch_target, effective_length


if __name__ == "__main__":
    vocab = Vocab(
        files=["./data/penn_train.txt", "./data/penn_valid.txt", "./data/penn_test.txt"],
    )
    print(len(vocab))
    penn_dataset = PennTreeBank(
        file_path="./data/penn_train.txt",
        max_length=30,
        vocab=vocab,
    )
    for i in range(10):
        print(penn_dataset.data[i])
    
    