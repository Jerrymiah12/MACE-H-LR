"""Data structures, graph datasets, preprocessing, and file-format I/O."""


def __getattr__(name):
    if name == "AijData":
        from .graph import AijData
        return AijData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AijData"]
