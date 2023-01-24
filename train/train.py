"""
Module for training RNN LM
"""
import torch
import torch.nn as nn

from data.data_utils import char_idx_to_str


def train_epoch(model, loss_fn, opt, x, y):
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


def train(model, X, Y, epochs=100, lr=0.01):
    '''
    model: torch nn.Module
    X: input data   (batch, seq_len)
    Y: target data  (batch, seq_len)
    '''
    #loss
    xe_loss = nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs+1):
        # TODO: sample for data loader to create batches
        loss = train_epoch(model, xe_loss, optimizer, X, Y)
        if epoch % 1000 == 0:
            y_ans = test_batch(model, X)
            print(f'Epoch: {epoch} | Loss: {loss}')
            print(y_ans)
            print('-----------------')


def test_batch(model, X):
    model.eval()
    y_hat = model(X)
    y_pred = y_hat.argmax(dim=-1).tolist()
    y_ans = []
    for ys in y_pred:
        yl = char_idx_to_str(ys)
        y_ans.append(''.join(yl))
    return y_ans
