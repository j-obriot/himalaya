"""The "torch_mps" GPU backend, based on PyTorch.

To use this backend, call ``himalaya.backend.set_backend("torch_mps")`` and
set the environment variable PYTORCH_ENABLE_MPS_FALLBACK=1.
"""
from .torch import *  # noqa
import torch

if not torch.backends.mps.is_available():
    import sys
    if "pytest" in sys.modules:  # if run through pytest
        import pytest
        pytest.skip("PyTorch with MPS is not available.")
    raise RuntimeError("PyTorch with MPS is not available.")

from ._utils import _dtype_to_str
from ._utils import warn_if_not_float32
from ._utils import _add_error_message

###############################################################################

name = "torch_mps"


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to("mps")


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs).to("mps")


def asarray(x, dtype=None, device="mps"):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        if hasattr(x, "dtype") and hasattr(x.dtype, "name"):
            dtype = x.dtype.name
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = getattr(torch, dtype)
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = "mps"
    try:
        tensor = torch.as_tensor(x, dtype=dtype, device=device)
    except Exception:
        import numpy as np
        array = np.asarray(x, dtype=_dtype_to_str(dtype))
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
    return tensor


def check_arrays(*all_inputs):
    """Change all inputs into Tensors (or list of Tensors) using the same
    precision and device as the first one. Some tensors can be None.
    """
    all_tensors = []
    all_tensors.append(asarray(all_inputs[0]))
    dtype = all_tensors[0].dtype
    warn_if_not_float32(dtype)
    device = all_tensors[0].device
    for tensor in all_inputs[1:]:
        if tensor is None:
            pass
        elif isinstance(tensor, list):
            tensor = [asarray(tt, dtype=dtype, device=device) for tt in tensor]
        else:
            tensor = asarray(tensor, dtype=dtype, device=device)
        all_tensors.append(tensor)
    return all_tensors


def zeros(shape, dtype="float32", device="mps"):
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def to_cpu(array):
    return array.cpu()


def to_gpu(array, device="mps"):
    return asarray(array, device=device)

# eigh and svd fallback to cpu
def svd(*a):
    a = list(a)
    for i in range(len(a)):
        if type(a[i]) == torch.Tensor:
            a[i] = a[i].cpu()
    U, s, V = torch.linalg.svd(*a)
    return U.to('mps'), s.to('mps'), V.to('mps')

def _eigh(*a):
    a = list(a)
    for i in range(len(a)):
        if type(a[i]) == torch.Tensor:
            a[i] = a[i].cpu()
    # eigenvalues, V = torch.linalg.eigh(*a)
    eigenvalues, V = torch.linalg.eigh(*a)
    return eigenvalues.to('mps'), V.to('mps')

eigh = _add_error_message(
        _eigh,
        msg=(f"The eigenvalues decomposition failed on backend {name}. You may"
             " try using `diagonalize_method='svd'`, or `solver_params={"
             "'diagonalize_method': 'svd'}` if called through the class API."))
