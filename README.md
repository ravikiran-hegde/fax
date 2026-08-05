# faxsec

`faxsec` packages the radiation-transfer and gas-optics utilities in this repository so they can be installed and imported from other directories.

## Install

```bash
pip install -e .
```

## Import

The legacy `model` package remains available after installation, and a `faxsec.model` compatibility namespace is provided for new code.

```python
from model.gas_optics import GasOptics
from faxsec.model import constants
```

## Examples

The scripts in `examples/` now rely on the installed package layout instead of inserting the repository root into `sys.path`. Run them after installing the project, for example:

```bash
python examples/test_sw.py
```

The examples still read and write datasets under `data/` relative to the repository.