from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from xarray import DataArray

ARRAYLIKE = ArrayLike | DataArray


@dataclass
class AbsorberConfig:
    species: str = ""
    frequency_grid: ARRAYLIKE = (1012.2305,)


class SingleSpeciesModel(ABC):
    """Abstract base class for species cross-section models."""

    config: Any

    def __init__(self, config: AbsorberConfig):
        self.config = config

    @abstractmethod
    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray:
        """Return cross-section matrix with shape (levels, frequency)."""
        ...

    def cross_section_from_atmds(self, atmosphere_ds: xr.Dataset) -> xr.DataArray:
        xsec = xr.apply_ufunc(
            self.cross_section,
            atmosphere_ds["pressure"],
            atmosphere_ds["temperature"],
            atmosphere_ds["vmr"].sel(species=self.config.species),
            input_core_dims=[[], [], []],  # all are (levels,)
            output_core_dims=[
                [
                    "frequency",
                ]
            ],  # output adds frequency dim
            dask="parallelized",  # works with dask arrays too
            output_dtypes=[float],
        )
        return xsec.assign_coords(frequency=self.config.frequency_grid).expand_dims(
            species=[str(self.config.species)], axis=0
        )

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Return the class name of the model."""
        return self.__class__.__name__


class SavableModel(ABC):
    """Abstract base class for models that can be saved to disk."""

    config: Any

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> SavableModel:
        """Create an instance of the model from an xarray Dataset."""
        ...

    @abstractmethod
    def to_dataset(self) -> xr.Dataset:
        """Convert the model to an xarray Dataset."""
        ...

    @abstractmethod
    def save_data(self, path: str | Path) -> None:
        """Save the model to disk."""
        ...

    @abstractmethod
    def load_data(self, path: str | Path) -> None:
        """Load the model from disk."""
        ...

    @property
    @abstractmethod
    def file_name(self) -> str:
        """Return the default file name for saving the model."""
        ...
        return f"{self.config.species}_{self.class_name}.nc"

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Return the class name of the model."""
        return self.__class__.__name__

    def save(self, path: Optional[str | Path] = None) -> None:
        """Save the model to disk."""
        if path is None:
            path = self.file_name
        self.save_data(path)
        self.save_config(path)

    def load(self, path: Optional[str | Path] = None) -> None:
        """Load the model from disk."""
        if path is None:
            path = self.file_name
        self.load_data(path)
        self.load_config(path)

    def save_config(self, path: str | Path) -> None:
        """Save the model configuration to disk."""
        config_path = Path(path) / f"{self.file_name}_config.json"
        with open(config_path, "w") as f:
            f.write(json.dumps(dataclasses.asdict(self.config), indent=4))

    def load_config(self, path: str | Path) -> None:
        """Load the model configuration from disk and load into its config type."""
        config_path = Path(path) / f"{self.file_name}_config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        self.config = self.config.__class__(**config_dict)
