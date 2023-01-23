import torch
import torch.nn as nn

class LSTM_CharLM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128) -> None:
        super().__init__()

        self.input_dim = 1
        self.hidden_dim = hidden_dim
        self.output_dim = 26

        self.hidden = nn.LSTM(
            self.input_dim, 
            self.hidden_dim,
            dropout=.5
            )

        # self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        hidden_op, (h_n, c_n) = self.hidden(x)
        logits = self.output(hidden_op)
        return logits


if __name__ == "__main__":
    m = LSTM_CharLM(1, 4)
    x = "hello world"
    x_vec = torch.Tensor([[ord(c)] for c in x])
    op = m(x_vec)
    print(op)
    
