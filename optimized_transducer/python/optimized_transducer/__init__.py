import torch

from .torch_version import optimized_transducer_torch_version

if torch.__version__.split("+")[0] != optimized_transducer_torch_version.split("+")[0]:
    raise ImportError(
        f"optimized_transducer was built using PyTorch {optimized_transducer_torch_version}\n"
        f"But you are using PyTorch {torch.__version__} to run it"
    )

from .transducer_loss import TransducerLoss, transducer_loss  # noqa
