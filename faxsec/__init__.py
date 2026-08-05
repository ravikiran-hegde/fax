"""Top-level package for faxsec."""

__all__ = ["model"]


def __getattr__(name: str):
    if name == "model":
        from . import model as model_module

        return model_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")