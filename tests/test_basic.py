import torch
from better_lstm import LSTM

def test_run_standard():
    lstm = LSTM(5, 10)
    x = torch.rand(3, 4, 5)
    a, (hn, cn) = lstm(x)
    assert a.size(0) == x.size(0)
    assert a.size(1) == x.size(1)
    assert a.size(2) == 10
