# char-rnn
Character-level Language Model built with RNN

# References
TODO fill out


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







