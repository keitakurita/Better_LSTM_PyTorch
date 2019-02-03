Better LSTM PyTorch
+++++++++++++++++++
An LSTM that incorporates best practices, designed to be fully compatible with the PyTorch LSTM API.
Implements the following best practices:
- Weight dropout
- Variational dropout in input and output layers
- Forget bias initialization to 1

These best practices are based on the following papers:
`A Theoretically Grounded Application of Dropout in Recurrent Neural Networks <https://arxiv.org/abs/1512.05287>`_
`Regularizing and Optimizing LSTM Language Models <https://arxiv.org/abs/1708.02182>`_
`An Empirical Exploration of Recurrent Network Architectures <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`

This code is heavily based on the code from this repository: most of the credit for this work goes to the authors.
(All I have done is update the code for PyTorch version 1.0 and repackage it).


Installation
============
Install via pip.

`$ pip install .`

Requires PyTorch version 1.0 or higher.


Usage
=====

.. code-block:: python

  >>> from better_lstm import LSTM
  >>> lstm = LSTM(100, 20, dropoutw=0.2)
