import pytest
import torch

from conv_lstm import ConvLSTM


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("input_dim", [5, 10])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("return_all_layers", [True, False])
def test_conv_lstm(
    kernel_size: int, input_dim: int, num_layers: int, return_all_layers: bool
) -> None:
    conv_lstm = ConvLSTM(
        input_dim=input_dim,
        hidden_dim=10,
        kernel_size=kernel_size,
        num_layers=num_layers,
        return_all_layers=return_all_layers,
    )
    input_tensor = torch.randn(64, 9, input_dim, 25, 25)
    out, states = conv_lstm(input_tensor)
    assert out is not None
    assert isinstance(out, list)
    if return_all_layers:
        assert len(out) == num_layers
    else:
        assert len(out) == 1
    for i in range(min(1, num_layers)):
        assert out[i].shape == torch.Size([64, 9, 10, 25, 25])


@pytest.mark.parametrize("return_all_layers", [True, False])
def test_conv_lstm_reduction(return_all_layers: bool) -> None:
    conv_lstm = ConvLSTM(
        input_dim=2,
        hidden_dim=(8, 4, 2),
        kernel_size=(5, 7, 9),
        num_layers=3,
        return_all_layers=return_all_layers,
    )
    input_tensor = torch.randn(64, 9, 2, 50, 50)
    out, states = conv_lstm(input_tensor)
    assert out is not None
    if return_all_layers:
        assert len(out) == 3
        assert len(states) == 3
        assert out[0].shape == torch.Size([64, 9, 8, 50, 50])
        assert out[1].shape == torch.Size([64, 9, 4, 50, 50])
        assert out[2].shape == torch.Size([64, 9, 2, 50, 50])
    else:
        assert len(out) == 1
        assert len(states) == 1
        assert len(states[0]) == 2
        assert out[0].shape == torch.Size([64, 9, 2, 50, 50])
        h, c = states[0]
        assert h.shape == torch.Size([64, 2, 50, 50])
        assert c.shape == torch.Size([64, 2, 50, 50])
