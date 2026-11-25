"""faxsec: functional forms for gas absorption cross-sections (FAX)."""

from importlib.metadata import PackageNotFoundError, version

from faxsec.arts import ARTSAbsorber
from faxsec.continuum import H2OContinuum
from faxsec.functional import FunctionalAbsorber, NoLogFunctionalAbsorber
from faxsec.gas_optics import GasOptics, load_gas_optics, trained_path
from faxsec.log_config import setup_logging
from faxsec.xfit import CrossFitAbsorber

try:
    __version__ = version("faxsec")
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
