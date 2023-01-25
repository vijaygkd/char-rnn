# char-rnn
Character-level Language Model built with RNN


# Observations / Considerations

* In the training script, the output char is selected with argmax. This causes the model to always select the same char conditioned for pretext. 
    * Maybe sampling from output distribution can be a good way to allow model to learn other output possiblities!? 
    * How is this done by others?
    * Or a larger dataset would help?

* Sampling tokens from distribution auto-regressively.
    * Basic version done.!

* Truncated BPTT and how to implement it?
    * Implemented by creating input sequences of length max_seq_len e.g. 100

* Measure perpexity

* Model
    * Add multiple hidden layers
    * Add dropout

* Larger corpus
    * Model train VERY SLOW on larger >1MB dataset :(
    * Try GPU training







