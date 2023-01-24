# char-rnn
Character-level Language Model built with RNN


# Observations / Considerations

* In the training script, the output char is selected with argmax. This causes the model to always select the same char conditioned for pretext. 
    * Maybe sampling from output distribution can be a good way to allow model to learn other output possiblities!? 
    * How is this done by others?
    * Or a larger dataset would help?

* Sampling tokens from distribution auto-regressively.

* Truncated BPTT and how to implement it?

* Measure perpexity

* Experiment with larger datasets





