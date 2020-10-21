from larq_zoo import literature, sota
from larq_zoo.core.utils import decode_predictions
from larq_zoo.training import datasets
from larq_zoo.training.data import preprocess_input

try:
    from importlib import metadata  # type: ignore
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata  # type: ignore

__version__ = metadata.version("larq_zoo")

__all__ = ["datasets", "decode_predictions", "preprocess_input", "literature", "sota"]
