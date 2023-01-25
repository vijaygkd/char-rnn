"""
Module for training RNN LM
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from data.data_utils import char_idx_to_str


def train_batch(model, loss_fn, opt, x, y):
    model.train()
    opt.zero_grad()
    y_hat = model(x)
    y_hat_perm = y_hat.permute((0,2,1)) 
    # permute dims so that output is (batch, no_classes, seq_len)
    # this ^ order is required by cross-entropy loss class
    loss = loss_fn(y_hat_perm, y)
    loss.backward()
    opt.step()
    return loss


def test_batch(model, loss_fn, x, y):
    model.eval()
    y_hat = model(x)
    y_hat_perm = y_hat.permute((0,2,1)) 
    # permute dims so that output is (batch, no_classes, seq_len)
    # this ^ order is required by cross-entropy loss class
    loss = loss_fn(y_hat_perm, y)
    return loss


def test_batch_gen_text(model, X):
    model.eval()
    y_hat = model(X)
    y_pred = y_hat.argmax(dim=-1).tolist()
    y_ans = []
    for y_idx in y_pred:
        yl = char_idx_to_str(y_idx)
        y_ans.append(yl)
    return y_ans


def train(model, train_dataloader, val_dataloader=None, epochs=1000, lr=0.01):
    '''
    model: torch nn.Module
    X: input data   (batch, seq_len)
    Y: target data  (batch, seq_len)
    '''
    #loss
    xe_loss = nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs+1), leave=False):
        train_loss = []
        for i, (X, Y) in enumerate(train_dataloader):
            loss = train_batch(model, xe_loss, optimizer, X, Y)
            train_loss.append(loss.item())
            
        if epoch % 100 == 0:
            print('-----------------')
            print(f'Epoch: {epoch} | Loss: {np.mean(train_loss)}')
            if val_dataloader is not None:
                val_loss = []
                for X_val, Y_val in val_dataloader:
                    loss = test_batch(model, xe_loss, X_val, Y_val)
                    val_loss.append(loss.item())

                    # y_test = test_batch(model, X_test)
                    # for y_str in y_test: print(y_str)

                print(f'Epoch: {epoch} | Val Loss: {np.mean(val_loss)}')
