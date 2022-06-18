'''
@ Contributor: Nayoung-Oh

'''

import argparse
from trainer import Trainer
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='wikilarge')
    parser.add_argument('--model', type=str, default='feature')
    parser.add_argument('--loss', type=str, default='weighted')
    parser.add_argument('--epoch', type=int, default=3)

    args = parser.parse_args()

    variant = vars(args)

    trainer = Trainer(variant["data"], variant["model"], variant["loss"])

    NUM_EPOCHS = variant["epoch"]
    trainer.transformer.load_state_dict(torch.load("log_1_val_1.328.pth"))
    min_val_loss = 10
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.evaluate(epoch)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"))
        if epoch % 1 == 0 or (epoch > 10 and val_loss < min_val_loss):
            print("save model")
            torch.save(trainer.transformer.state_dict(), f"log_{epoch}_val_{val_loss:.3f}.pth")
        min_val_loss = min(val_loss, min_val_loss)
