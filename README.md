# char-rnn
Character-level Language Model built with RNN

# References
* Inspiration of this repo, Andrej Karpathy's blog [The Unreasonable Effectiveness of Recurrent Neural Networks]http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* Karpathy's repo [char-rnn](https://github.com/karpathy/char-rnn)
* Stanford NLP [Lecture 6 - RNN & LSTM](https://www.youtube.com/watch?v=0LixFSa7yts&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ&index=7)

# Observations / Considerations

* In the training script, the output char is selected with argmax. This causes the model to always select the same char conditioned for pretext. 
    * Maybe sampling from output distribution can be a good way to allow model to learn other output possiblities!? 
    * How is this done by others?
    * Or a larger dataset would help?

* Sampling tokens from distribution auto-regressively.
    * Basic version done.!

* Dropout
    * Emperically, it does seem to prevent overfitting of the model. 
    * Val loss keep decreasing with decrease in train loss. 
    * On small datasets, the val loss starts to increase soon after.
    * Tried dropout=0.5

* Truncated BPTT and how to implement it?
    * Implemented by creating input sequences of length max_seq_len e.g. 100

* Measure perpexity
    * Perplexity = exp(cross_entropy_loss)

* Larger corpus
    * Model train VERY SLOW on larger >1MB dataset :(
    * Try GPU training







