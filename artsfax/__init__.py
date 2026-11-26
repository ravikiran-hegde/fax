"""artsfax: functional forms for gas absorption cross-sections (FAX)."""

from importlib.metadata import PackageNotFoundError, version

from artsfax.arts import ARTSAbsorber
from artsfax.continuum import H2OContinuum
from artsfax.functional import FunctionalAbsorber, NoLogFunctionalAbsorber
from artsfax.gas_optics import GasOptics, load_gas_optics, trained_path
from artsfax.log_config import setup_logging
from artsfax.xfit import CrossFitAbsorber

try:
    __version__ = version("artsfax")
except PackageNotFoundError:  # not installed
    __version__ = "unknown"

__all__ = [
    "ARTSAbsorber",
    "CrossFitAbsorber",
    "FunctionalAbsorber",
    "GasOptics",
    "H2OContinuum",
    "NoLogFunctionalAbsorber",
    "load_gas_optics",
    "setup_logging",
    "trained_path",
]
