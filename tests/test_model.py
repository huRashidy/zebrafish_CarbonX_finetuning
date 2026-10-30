import pytest
import torch
import torch.nn as nn
from src.model import get_layer_device_mapping


class DummyModelWithDeviceMap:
    def __init__(self, device_map):
        self.hf_device_map = device_map


def test_get_layer_device_mapping():
    mock_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.19": 0,
        "model.layers.20": 1,
        "model.layers.39": 1,
        "model.norm": 1,
        "lm_head": 1,
    }
    dummy = DummyModelWithDeviceMap(mock_map)
    layer_mapping = get_layer_device_mapping(dummy)
    assert 0 in layer_mapping
    assert 19 in layer_mapping
    assert 20 in layer_mapping
    assert 39 in layer_mapping
    assert layer_mapping[0] == 0
    assert layer_mapping[19] == 0
    assert layer_mapping[20] == 1
    assert layer_mapping[39] == 1
    assert "model.norm" not in layer_mapping


def test_target_gpu_layer_selection():
    mock_map = {
        f"model.layers.{i}": 0 if i < 20 else 1
        for i in range(40)
    }
    dummy = DummyModelWithDeviceMap(mock_map)
    layer_mapping = get_layer_device_mapping(dummy)
    target_gpu = 1
    target_layers = [idx for idx, dev in layer_mapping.items() if dev == target_gpu]
    assert len(target_layers) == 20
    assert min(target_layers) == 20
    assert max(target_layers) == 39
