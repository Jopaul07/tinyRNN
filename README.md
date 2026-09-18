# tinyRNN

Building recurrent neural networks from scratch in PyTorch — a progression from
vanilla RNNs to LSTMs to LayerNorm-stabilized LSTMs.

## Notebooks

### [build_tinyRNN_vanillaRNN.ipynb](build_tinyRNN_vanillaRNN.ipynb)
Builds a vanilla RNN from scratch, then an LSTM with explicit gates and cell
state, training both on daily gold prices. Validates the LSTM bit-exact against
`nn.LSTM` and explains why predicting efficient-market prices rarely beats the
naive "do nothing" forecaster.

### [build_tinyRNN_LSTM.ipynb](build_tinyRNN_LSTM.ipynb)
Implements an LSTM cell gate-by-gate to solve vanishing gradients, trains a
makemore-style character-level name generator, verifies it against PyTorch's
`nn.LSTM`, and visualizes gate activations, cell-state evolution, and
temperature-controlled name sampling.

### [build_tinyRNN_LayerNorm.ipynb](build_tinyRNN_LayerNorm.ipynb)
Implements LayerNorm from scratch and wires it into the LSTM's gate
pre-activations. Contrasts LayerNorm vs BatchNorm for RNNs and runs side-by-side
experiments showing LayerNorm enables faster, more stable training at higher
learning rates.
