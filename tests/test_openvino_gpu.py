"""Tests for whisper.unsupported_gpu_reason (the openvino Core is faked)."""

import pytest

from whisper_ov import whisper


class _FakeCore:
    def __init__(self, devices):
        self.devices = devices

    def get_property(self, device, name):
        if device not in self.devices:
            raise RuntimeError(f"[GPU] Can't get {name}: device {device} not found")
        return self.devices[device][name]


def _gpu(full_name, vendor):
    return {"FULL_DEVICE_NAME": full_name, "DEVICE_ARCHITECTURE": f"GPU: vendor={vendor} arch=v5.0.0"}


CORE = _FakeCore(
    {
        "GPU.0": _gpu("Intel(R) Iris(R) Xe Graphics (iGPU)", "0x8086"),
        "GPU.1": _gpu("NVIDIA GeForce GTX 950M (dGPU)", "0x10de"),
        "GPU.2": _gpu("AMD Radeon RX 6600 (dGPU)", "0x1002"),
        "CPU": {"FULL_DEVICE_NAME": "Intel(R) Core(TM) i7-6500U", "DEVICE_ARCHITECTURE": "intel64"},
    }
)


@pytest.mark.parametrize("device", ["GPU.0", "CPU", "NPU", "AUTO", "GPU.9"])
def test_supported_or_not_a_gpu(device):
    assert whisper.unsupported_gpu_reason(device, CORE) is None


def test_nvidia_gpu_suggests_cuda():
    reason = whisper.unsupported_gpu_reason("GPU.1", CORE)
    assert "NVIDIA GeForce GTX 950M" in reason
    assert "-d CUDA" in reason


def test_other_vendor_gpu_suggests_cpu():
    reason = whisper.unsupported_gpu_reason("GPU.2", CORE)
    assert "AMD Radeon" in reason
    assert "-d CUDA" not in reason
    assert "-d CPU" in reason
