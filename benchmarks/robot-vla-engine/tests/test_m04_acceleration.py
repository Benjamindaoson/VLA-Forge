import pytest
import torch

from robot_vla.m04_acceleration import validate_tensor_inputs


def test_graph_rejects_shape_or_dtype_change():
    old = [torch.zeros(1, 3)]
    validate_tensor_inputs(old, [torch.ones(1, 3)])
    with pytest.raises(ValueError, match="metadata"):
        validate_tensor_inputs(old, [torch.ones(2, 3)])
    with pytest.raises(ValueError, match="metadata"):
        validate_tensor_inputs(old, [torch.ones(1, 3, dtype=torch.float64)])


def test_graph_rejects_missing_inputs():
    with pytest.raises(ValueError, match="count"):
        validate_tensor_inputs([torch.zeros(1)], [])
