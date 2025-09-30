import os
from functools import lru_cache
from typing import NamedTuple
import torch


class DeviceInfo(NamedTuple):
    device: torch.device
    ngpu: int
    name: str
    backend: str  # "cuda" | "mps" | "cpu"

    def __str__(self):
        return f"DeviceInfo({self.device}, {self.ngpu}, {self.name})"

    def __repr__(self):
        return f"DeviceInfo({self.device}, {self.ngpu}, {self.name})"


def _env_requested_device() -> str | None:
    """
    Optional override via env var, e.g. TORCH_DEVICE=cuda:0 / mps / cpu
    """
    val = os.getenv("TORCH_DEVICE")
    return val.strip() if val else None


def _detect_device() -> DeviceInfo:
    # 1) explicit override
    override = _env_requested_device()
    if override:
        try:
            dev = torch.device(override)
            backend = dev.type
            if backend == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if backend == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            ngpu = (
                torch.cuda.device_count()
                if backend == "cuda"
                else (1 if backend == "mps" else 0)
            )
            name = (
                torch.cuda.get_device_name(dev.index or 0)
                if backend == "cuda"
                else "Apple Silicon (MPS)" if backend == "mps" else "CPU"
            )
            return DeviceInfo(dev, ngpu, name, backend)
        except Exception as e:
            raise RuntimeError(f"Invalid TORCH_DEVICE={override}: {e}")

    # 2) auto-detect (priority: MPS > CUDA > CPU) — adjust if you prefer CUDA first
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(torch.device("mps"), 1, "Apple Silicon (MPS)", "mps")

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count > 0 else "CUDA GPU"
        return DeviceInfo(torch.device("cuda:0"), count, name, "cuda")

    return DeviceInfo(torch.device("cpu"), 0, "CPU", "cpu")


@lru_cache(maxsize=1)
def get_device_info() -> DeviceInfo:
    return _detect_device()


# Eagerly evaluate once so you can do `from device_config import DEVICE`
DEVICE_INFO = get_device_info()
DEVICE: torch.device = DEVICE_INFO.device
NGPU: int = DEVICE_INFO.ngpu
DEVICE_NAME: str = DEVICE_INFO.name
BACKEND: str = DEVICE_INFO.backend


def is_cuda() -> bool:
    return BACKEND == "cuda"


def is_mps() -> bool:
    return BACKEND == "mps"


def is_cpu() -> bool:
    return BACKEND == "cpu"
