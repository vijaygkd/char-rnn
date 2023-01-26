import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from data.data_utils import process_text


def process_input_file(file_path, seq_len=50, batch_size=100, train_frac=0.95):
    ''' 
    Read text from input file and process it into batches with seq_len
    Returns train and val dataloader
    '''
    # read text file
    with open(file_path, 'r') as f:
        text = f.read()
    
    # split text in chuncks 
    p = batch_size * (seq_len) - 1  #start and end token for a batch
    n_chunks = len(text) // p
    # remove tail end of text to make even chunks
    if len(text) % p > 0:
        text = text[:p*n_chunks]

    X_all = []
    Y_all = []
    for i in range(n_chunks):
        text_chunk = text[i*p : (i+1)*p]
        x, y = process_text(text_chunk)
        X_all.append(x)
        Y_all.append(y)

    # break into seq_len samples
    X_all = torch.tensor(X_all).view(-1,seq_len)
    Y_all = torch.tensor(Y_all).view(-1,seq_len)
    dataset = TensorDataset(X_all, Y_all)
    train_dataset, val_dataset = random_split(dataset, [train_frac, 1-train_frac])
    
    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(f"Train: samples={len(train_dataset)} batches={len(train_dataloader)}" + 
          f" | Val: samples={len(val_dataset)} batches={len(val_dataloader)}")

    return train_dataloader, val_dataloader
