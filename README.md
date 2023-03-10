# char-rnn
Character-level Language Model built with RNN / LSTM

# References
* Inspiration of this repo, Andrej Karpathy's blog [The Unreasonable Effectiveness of Recurrent Neural Networks]http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* Karpathy's repo [char-rnn](https://github.com/karpathy/char-rnn)
* Stanford NLP [Lecture 6 - RNN & LSTM](https://www.youtube.com/watch?v=0LixFSa7yts&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ&index=7)

# Usage

**Requirements:**
* `python 3.9`
* `pip`
* `poetry`

You can start training and using the model with less than 10 lines of code. Sample models are built in jupyter notebooks using the API from the repo.


# How it works?

## Data
The input text data for training is saved as a `.txt` file in the directory `data_files/`. The directory contains a couple of sample datafiles, including randomly generated sentences about coffee and some samples from works by Shakespeare.

### Dataset Size
The coffee dataset is a good toy dataset for quick development. The model trains quickly on this dataset, but it will likely overfit it.

The Shakespeare dataset is an example of a medium-sized dataset, with a size of around 1MB. The dataset is large enough that it takes more than 24 hours for a large model to train for 1000 epochs.

### BYOD (Bring Your Own Data)
You can also use your own dataset by creating a `.txt` file of any length or size.

## Model
The Character-level language model is trained to predict the next character, given a sequence of preceding characters. For example, given the sequence "hell", the model tries to predict the character "o" to make the word "hello".

The model can read as input any character from the 128-character ASCII set. The model outputs probabilities of 128 characters for every location in the sequence.

The model is built using LSTM layers as the hidden layers. The API takes as input the following parameters for defining the network:

* `hidden_dim`: Number of units in the hidden LSTM layer
* `hidden_layers`: Number of hidden LSTM layers. 1-3 is optimal
* `dropout`: Dropout for hidden layers

For a single data sample or smaller coffee dataset, 1-layer with 128 nodes can easily overfit the training data. For a larger dataset, 2-3 layers with 256-512 units are required. This increases the required training time as well.

## Training
### Data Processing
Input data can contain any character from the `128-char ASCII set`, including alphanumeric characters, symbols, and indentation characters from the .txt file.

The dataloader class reads the input file and generates `train` and `validation` dataloaders based on the `train_frac` parameter. Each dataloader then provides data in batches of fixed sequence sentences. These parameters are defined by `batch_size` and `seq_len` parameters.

The labels corresponding to an input sequence are the next character in the sequence shifted by one place.

### Truncated Backpropagation Through Time
The training algorithm uses truncated BPTT with a sequence length of 100 for training the model. Truncated Backprop is implemented simply by slicing the input sequence in lengths as defined by the `seq_len` parameter.

Larger sequence length allows the model to learn longer dependencies in the data, but it also results in longer training time.

## Sampling the Model
New text can be generated from the model by sampling from the output probabilities and feeding the output of the model as the input at the next timestamp in an auto-regressive manner.

### Priming
You can also provide `context` to prime the model for generating new text conditioned on the context.


# Considerations

**Output Selection**
* The current training script uses argmax to select the output character, which causes the model to always select the same character conditioned on the pretext. To allow the model to learn other output possibilities, consider sampling from the output distribution as an alternative approach.

**Sampling Tokens**
* The model generates new text by sampling tokens from the distribution in an auto-regressive manner. To improve performance, consider implementing temperature for softmax and beam search algorithms.

**Overfitting Prevention**
* Dropout has been empirically observed to prevent overfitting of the model. However, when working with small datasets, the validation loss may start to increase soon after. A dropout rate of 0.5 has been tried in the past.

**Truncated Backpropagation Through Time (BPTT)**
* Truncated BPTT is implemented by creating input sequences of maximum length, e.g. 100. This is in line with the implementation used in Karpathy's repo, but alternative methods may also be considered.

**Language Model Perplexity**
* Perplexity = `exp(cross-entropy loss)`

**GPU Training**
* The current model may train very slowly on larger datasets (>1MB). To improve performance, consider training the model on a GPU.
