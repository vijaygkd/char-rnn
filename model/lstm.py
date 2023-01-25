import torch
import torch.nn as nn

class LSTM_CharLM(nn.Module):
    def __init__(self, hidden_dim=128, hidden_layers=1) -> None:
        super().__init__()

        self.char_vocab_size = self.output_dim = 128      # ascii char set
        self.embed_dim = 12
        self.hidden_dim = hidden_dim

        self.char_embedding = nn.Embedding(self.char_vocab_size, self.embed_dim)

        self.hidden = nn.LSTM(
            self.embed_dim, 
            self.hidden_dim,
            num_layers=hidden_layers,
            dropout=0.5,
            batch_first=True,
        )

        self.output = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        char_emb = self.char_embedding(x)
        hidden_op, (h_n, c_n) = self.hidden(char_emb)
        logits = self.output(hidden_op)
        return logits


if __name__ == "__main__":
    m = LSTM_CharLM(1, 4)
    x = "hello world"
    x_vec = torch.Tensor([[ord(c)] for c in x])
    op = m(x_vec)
    print(op)
