import torch
from better_lstm import LSTM, VariationalDropout

def is_close(x, val, thres=1e-5):
    return abs(x - val) < thres

def test_run_standard():
    lstm = LSTM(5, 10)
    x = torch.rand(3, 4, 5)
    a, (hn, cn) = lstm(x)
    assert a.size(0) == x.size(0)
    assert a.size(1) == x.size(1)
    assert a.size(2) == 10

def test_run_weight_dropout():
    lstm = LSTM(5, 10, dropoutw=0.2)
    x = torch.rand(3, 4, 5)
    a, (hn, cn) = lstm(x)

def test_run_input_output_dropout():
    lstm = LSTM(5, 10, dropouti=0.2, dropouto=0.5)
    x = torch.rand(3, 4, 5)
    a, (hn, cn) = lstm(x)

def test_variational_dropout():
    dr = 0.3
    seq_len = 10
    do = VariationalDropout(dr, batch_first=True)
    x = torch.ones(3, seq_len, 5)
    dropped_x = do(x)
    x = dropped_x.sum(1)
    for i in range(3):
        for j in range(5):
            assert is_close(x[i, j], 0) or is_close(x[i, j], seq_len / (1-dr)), f"Element {i},{j} was {x[i, j]}"
