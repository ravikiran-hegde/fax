# ARTS-FAX

Gas absorption cross-sections for radiative transfer from ARTS-FAX: functional forms
for absorption cross-sections. At each frequency, every gas uses a small
closed-form function of pressure and temperature, trained against ARTS
line-by-line calculations. A trained model is a few coefficients per frequency
and gas.

Trained models for the 64-point DDQ longwave and shortwave quadratures are in
`data/ff`.

## Install

```bash
git clone https://github.com/ravikiran-hegde/fax.git
cd fax
pip install -e .
```

Extras: `.[arts]` to train or compare against ARTS (pyarts3; developed with
3.0.0.dev10), `.[rte]` for the flux example (pyrte_rrtmgp), `.[dev]` for the
tests.

## Use

```python
import numpy as np
import xarray as xr

import artsfax

gas_optics = artsfax.load_gas_optics("LW")  # or "SW"

h2o = gas_optics.absorbers["H2O_ShiftedReciprocalLaurent_Rational"]
xsec = h2o.cross_section(
    pressure=np.array([5e4]), temperature=np.array([250.0]), vmr=np.array([5e-3])
)  # m2 per molecule, shape (levels, frequency)

atmosphere = xr.Dataset(
    {
        "pressure_layer": ("layer", [2e3, 3e4, 9e4]),  # Pa
        "temperature_layer": ("layer", [220.0, 240.0, 285.0]),  # K
        "N_per_m2_dry": ("layer", [4e27, 1.5e28, 3e28]),  # molecules per m2
        "H2O": ("layer", [5e-6, 5e-4, 1e-2]),  # volume mixing ratio
        "CO2": ("layer", [4.2e-4] * 3),
    }
)
tau = gas_optics.optical_depth_from_ds(atmosphere)  # species x layer x frequency
```

## Examples

- `examples/rte_example.py`: longwave and shortwave fluxes for the RFMIP,
  CKDMIP and RCE cases, with FAX, ARTS or RRTMGP gas optics.
  `python examples/rte_example.py --case rfmip --gas-optics fax`
- `examples/train_ddq.py`, `examples/train_fgrid.py`: train on a DDQ
  quadrature or on a dense frequency grid.
- `tools/cluster`: SLURM scripts for dense-grid training and line-by-line fluxes.
