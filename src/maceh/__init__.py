"""MACE-H-LR public API.

Imports are lazy so lightweight data/response utilities do not require the
optional neural-network stack merely to import :mod:`maceh`.
"""

from importlib import import_module

__version__ = "0.1.0"

_PUBLIC = {
    "AijData": ("maceh.data.graph", "AijData"),
    "Collater": ("maceh.graph", "Collater"),
    "get_graph": ("maceh.graph", "get_graph"),
    "load_orbital_types": ("maceh.graph", "load_orbital_types"),
    "Net": ("maceh.maceh", "Net"),
    "LossRecord": ("maceh.utils", "LossRecord"),
    "Rotate": ("maceh.e3modules", "Rotate"),
    "e3TensorDecomp": ("maceh.e3modules", "e3TensorDecomp"),
    "DeepHE3Kernel": ("maceh.kernel", "DeepHE3Kernel"),
    "testResultAnalyzer": ("maceh.analyzer", "testResultAnalyzer"),
}

__all__ = ["__version__", *_PUBLIC]


def __getattr__(name):
    try:
        module_name, attribute = _PUBLIC[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC))
