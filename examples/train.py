import os
import torch
from nplm.nplmodel import NPLModel
from nplm.models.lstm.lstm import LSTMModel
from nplm.data_loader.penn_loader import PennTreeBank, PennTreeBankLoader
from nplm.data_loader.vocab import Vocab
from nplm.criterion.train_criterion import CrossEntropyLoss, SimpleNCELoss
from nplm.criterion.eval_criterion import PPL, WER
import argparse

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
    
    if args.model_type == "lstm":
        lstm = LSTMModel(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        raise ValueError("Model type not supported.")
    if args.loss_type == "cross_entropy":
        criterion = CrossEntropyLoss()
    elif args.loss_type == "nce":
        criterion = SimpleNCELoss(vocab=vocab, k=args.k)
    else:
        raise ValueError("Loss type not supported.")
    
    nplmodel = NPLModel(
        logit_model=lstm,
        vocab=vocab,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    nplmodel.train(
        data_loader=train_dataloader,
        optimizer=torch.optim.SGD,
        criterion=criterion,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPLM")
    parser.add_argument("--model_type", type=str, default="lstm", help="Type of the model.")
    parser.add_argument("--loss_type", type=str, default="nce", help="Type of the loss function.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--k", type=int, default=1000, help="Number of noise samples.")
    args = parser.parse_args()
    
    main(args)