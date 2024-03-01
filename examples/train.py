import os
import torch
from nplm.nplmodel import NPLModel
from nplm.models.lstm.lstm import LSTMModel
from nplm.models.rnn.rnn import RNNModel
from nplm.data_loader.penn_loader import PennTreeBank, PennTreeBankLoader
from nplm.data_loader.vocab import Vocab
from nplm.criterion.train_criterion import CrossEntropyLoss, SimpleNCELoss, NCELoss
from nplm.criterion.eval_criterion import PPL, WER
import numpy as np
import argparse
import pickle

def main(args):
    vocab = Vocab(files=["../nplm/data_loader/data/penn_train.txt"])
    print("Length of the Vocabulary: ", len(vocab))
    train_dataset = PennTreeBank(
        file_path="../nplm/data_loader/data/penn_train.txt",
        vocab=vocab,
    )
    train_dataloader = PennTreeBankLoader(
        dataset=train_dataset,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataset = PennTreeBank(
        file_path="../nplm/data_loader/data/penn_valid.txt",
        vocab=vocab,
    )
    val_dataloader = PennTreeBankLoader(
        dataset=val_dataset,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "lstm":
        lstm = LSTMModel(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model_type == "rnn":
        lstm = RNNModel(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        raise ValueError("Model type not supported.")
    if args.loss_type == "ce":
        criterion = CrossEntropyLoss()
    elif args.loss_type == "simplence":
        criterion = SimpleNCELoss(num_class=len(vocab), k=args.k, device=device)
    elif args.loss_type == "nce":
        if args.noise_type == "unigram":
            word_freq = np.array(vocab.index2freq)
            word_distribution = torch.from_numpy(word_freq / len(vocab))
        elif args.noise_type == "uniform":
            word_distribution = None
        else:
            raise ValueError("Noise type not supported.")
        criterion = NCELoss(num_class=len(vocab), k=args.k, noise_distribution=word_distribution, device=device)
    else:
        raise ValueError("Loss type not supported.")
    
    nplmodel = NPLModel(
        logit_model=lstm,
        vocab=vocab,
        device=device,
    )
    
    nplmodel.train(
        data_loader=train_dataloader,
        optimizer=torch.optim.Adam,
        criterion=criterion,
        epochs=args.epochs,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        val_data_loader=val_dataloader,
    )
    
    test_dataset = PennTreeBank(
        file_path="../nplm/data_loader/data/penn_test.txt",
        vocab=vocab,
    )
    test_dataloader = PennTreeBankLoader(
        dataset=test_dataset,
        vocab=vocab,
        batch_size=args.batch_size,
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
    
    print("================================Show Some Generated Sentences================================")
    output_string = nplmodel.generate(start_token="we", max_length=100)
    print(output_string)
    output_string = nplmodel.generate(start_token="he", max_length=100)
    print(output_string)
    
    ## Save the model
    os.makedirs("./save", exist_ok=True)
    nplmodel.save(path="./save")
    args_dict = vars(args)
    with open("./save/args.pkl", "wb") as f:
        pickle.dump(args_dict, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPLM")
    parser.add_argument("--model_type", type=str, default="lstm", help="Type of the model.")
    parser.add_argument("--loss_type", type=str, default="nce", help="Type of the loss function.")
    parser.add_argument("--noise_type", type=str, default="unigram", help="Type of the nce noise distribution.")
    parser.add_argument("--lr_scheduler", type=bool, default=True, help="Use learning rate scheduler.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--k", type=int, default=100, help="Number of noise samples.")
    args = parser.parse_args()
    
    main(args)