import os
import pickle

UNK = '<unk>'  # unknown word
BOS = '<s>'  # sentence start
EOS = '</s>'  # sentence end

class Vocab(object):
    """
    Vocabulary class.
    Args:
        
    """
    def __init__(self, files: list, max_vocab_size: int = None, min_freq: int = 1):
        counter = {}
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    counter[BOS] = counter.get(BOS, 0) + 1
                    counter[EOS] = counter.get(EOS, 0) + 1
                    words = line.split()
                    for word in words:
                        if word in counter:
                            counter[word] += 1
                        else:
                            counter[word] = 1
        # sort by frequency, then alphabetically
        words_and_freq = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words_and_freq.sort(key=lambda tup: tup[1], reverse=True)
        
        unk_freq = 0
        self.index2word = [UNK, BOS, EOS]
        for word, freq in words_and_freq:
            if freq < min_freq or (max_vocab_size is not None and len(self.index2word) >= max_vocab_size):
                unk_freq += freq
            else:
                if word not in [UNK, BOS, EOS]:
                    self.index2word.append(word)
        self.word2index = {word: idx for idx, word in enumerate(self.index2word)}
        self.index2freq = [counter.get(word, unk_freq) for word in self.index2word]
        
        assert len(self.index2word) == len(self.word2index) == len(self.index2freq)
        
    def __eq__(self, other: object) -> bool:
        if self.word2index != other.word2index:
            return False
        if self.index2word != other.index2word:
            return False
        if self.index2freq != other.index2freq:
            return False
        return True
    
    def __len__(self) -> int:
        return len(self.index2word)
    
    def query(self, word: str) -> int:
        """
        Query the index of a word.
        Args:
            word (str): Word to query.
        """
        if word in self.word2index:
            return self.word2index[word]
        else:
            return -1
    
    def save(self, path: str):
        """
        Save the vocabulary.
        Args:
            path (str): File path to save the vocabulary.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'Vocab':
        """
        Load the vocabulary.
        Args:
            path (str): File path to load the vocabulary.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
        
        
        
        