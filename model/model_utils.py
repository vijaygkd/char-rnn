import numpy as np
import torch

from data.data_utils import str_to_ids, char_idx_to_str


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def sample_token(model, context='', max_len=100):
    model.eval()
    X_context = torch.tensor(str_to_ids("$"+context))
    for i in range(max_len):
        y_logit = model(X_context)[-1]      # last output char distribution
        y_logit = y_logit.detach().numpy()
        y_prob = softmax(y_logit)
        # sample value from distribution
        ch_sample = np.random.choice(len(y_prob), size=1, p=y_prob)
        # auto-regress the sampled character for next input
        X_context = torch.cat((X_context, torch.tensor(ch_sample)))
        
    output_str = char_idx_to_str(X_context.tolist())
    return output_str
