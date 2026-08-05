"""Compatibility layer exposing the legacy model package as faxsec.model."""

from importlib import import_module as _import_module
import sys as _sys

_SUBMODULES = (
    "abstract_class",
    "arts",
    "constants",
    "continuum",
    "forms",
    "functional",
    "gas_optics",
    "utils",
    "xfit",
)

for _name in _SUBMODULES:
    _module = _import_module(f"model.{_name}")
    _sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

__all__ = list(_SUBMODULES)