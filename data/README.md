# Data

| folder | contents | source |
|---|---|---|
| `ff/` | trained gas optics: DDQ quadratures (`gas_optics_DDQ_*`) and dense grid (`gas_optics_Highres_*`) | this repository |
| `ddq/` | DDQ quadratures: frequencies and weights | [pczarnecki/ddq-data-paper](https://github.com/pczarnecki/ddq-data-paper) |
| `continuum/` | MT_CKD 4.3 water vapour continuum (`...-mt-ckd400.nc` is 4.0) | [AER-RC/MT_CKD_H2O](https://github.com/AER-RC/MT_CKD_H2O) |
| `halocarbon/` | XFIT halocarbon cross-section fits | [ARTS XML data](https://github.com/atmtools/arts-xml-data) |
| `solar_spectra/` | solar spectrum, July 2008 | [ARTS XML data](https://github.com/atmtools/arts-xml-data) |
| `rte_examples/` | RFMIP, CKDMIP and RCE atmospheres | [earth-system-radiation/rte-examples](https://github.com/earth-system-radiation/rte-examples) |

The trained models record how they were trained in their attributes:
`training_config` and `config_detail` at the root, the forms in each group.
