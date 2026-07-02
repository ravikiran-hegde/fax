# %%
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.utils import hz_to_kayser


def clear_all_attrs(ds):
    ds = ds.copy()
    ds.attrs.clear()
    for var in ds.variables:
        ds[var].attrs.clear()
    return ds


# -----------------------------------------------------------------------------
# %% Order and renames
# -----------------------------------------------------------------------------

TERM_NAMES = ["p00", "p10", "p20", "p01"]

LINE_RENAME = {
    "xsec0": "fax_sigma0",
    "p_order": "fax_p_nterms",
    "ref_pressure": "fax_p0",
    "ref_temperature": "fax_T0",
    "ref_vmr": "fax_vmr0",
    "self_scaling": "fax_S",
    "pressure_coeffs": "fax_c",
    "species": "fax_species_names",
}

CONT_RENAME = {
    "ref_pressure": "mtckd_p0",
    "ref_temperature": "mtckd_T0",
    "species": "mtckd_species_names",
    "self_absco_ref": "mtckd_cself",
    "self_texp": "mtckd_n",
    "for_absco_ref": "mtckd_cfrgn",
}

TRANSPOSE_ORDER = (
    "nu",
    "fax_species_names",
    "fax_p_nterms",
    "fax_t_order",
    "xsec_species_names",
    "xsec_nterms",
    "mtckd_species_names",
)

LW_ORDER = [
    "nu",
    "weights",
    "fax_nspecies",
    "fax_species_names",
    "fax_p0",
    "fax_T0",
    "fax_S",
    "fax_sigma0",
    "fax_p_nterms",
    "fax_c",
    "fax_t_order",
    "fax_a",
    "fax_b",
    "xsec_nspecies",
    "xsec_species_names",
    "xsec_nterms",
    "xsec_p",
    "mtckd_nspecies",
    "mtckd_species_names",
    "mtckd_p0",
    "mtckd_T0",
    "mtckd_cself",
    "mtckd_n",
    "mtckd_cfrgn",
]

SW_ORDER = [
    "nu",
    "weights",
    "solar_spectral_irradiance",
    *LW_ORDER[2:],
    "rayleigh_xsec",
]


# =============================================================================
# %%Longwave
# =============================================================================

data_lw = xr.open_datatree("../data/ff/test_3_lw.nc").copy()
# ---- Hinge rational ---------------------------------------------------------

lines = data_lw["Hinge_Rational"].to_dataset().rename(LINE_RENAME)

lines["fax_p_nterms"] = ("fax_p_nterms", range(lines.sizes["fax_p_nterms"]))

lines["fax_a"] = (
    lines["temperature_coeffs"]
    .isel(t_order=[0, 1, 2])
    .rename({"t_order": "fax_t_order"})
    .assign_coords(fax_t_order=[0, 1, 2])
)

ones = xr.ones_like(lines["temperature_coeffs"].isel(t_order=0, drop=True))
ones = ones.expand_dims(fax_t_order=[0])

rest = (
    lines["temperature_coeffs"]
    .isel(t_order=[3, 4])
    .rename({"t_order": "fax_t_order"})
    .assign_coords(fax_t_order=[1, 2])
)

lines["fax_b"] = xr.concat([ones, rest], dim="fax_t_order")

lines = lines.drop_vars(["temperature_coeffs", "t_order", "fax_vmr0"])
lines["fax_species_names"] = lines["fax_species_names"].str.lower()
# ---- MTCKD ------------------------------------------------------------------

cont = data_lw["both_continuum_MT_CKD_4_3"].to_dataset().rename(CONT_RENAME)
cont["mtckd_species_names"] = cont["mtckd_species_names"].str.lower()
# ---- XFIT -------------------------------------------------------------------

xsec = (
    xr.concat(
        [data_lw["XFIT"].to_dataset()[name] for name in TERM_NAMES],
        dim="xsec_nterms",
    )
    .assign_coords(xsec_nterms=range(4))
    .to_dataset()
    .rename(
        {
            "species": "xsec_species_names",
            "p00": "xsec_p",
        }
    )
)
xsec["xsec_species_names"] = xsec["xsec_species_names"].str.lower()
# ---- Merge ------------------------------------------------------------------

gas_optics_lw = xr.merge([lines, cont, xsec])

gas_optics_lw["frequency"] = hz_to_kayser(gas_optics_lw["frequency"])
gas_optics_lw = gas_optics_lw.rename({"frequency": "nu"})

gas_optics_lw["weights"] = (
    "nu",
    hz_to_kayser(data_lw["DDQ"]["weights_hz"].values),
)

gas_optics_lw = clear_all_attrs(gas_optics_lw)
gas_optics_lw = gas_optics_lw.transpose(*TRANSPOSE_ORDER)

gas_optics_lw = gas_optics_lw.assign_coords(
    fax_nspecies=("fax_species_names", range(gas_optics_lw.sizes["fax_species_names"])),
    xsec_nspecies=(
        "xsec_species_names",
        range(gas_optics_lw.sizes["xsec_species_names"]),
    ),
    mtckd_nspecies=(
        "mtckd_species_names",
        range(gas_optics_lw.sizes["mtckd_species_names"]),
    ),
)

gas_optics_lw = (
    gas_optics_lw.swap_dims({"fax_species_names": "fax_nspecies"})
    .swap_dims({"xsec_species_names": "xsec_nspecies"})
    .swap_dims({"mtckd_species_names": "mtckd_nspecies"})
    .reset_coords(["fax_species_names", "xsec_species_names", "mtckd_species_names"])
)
gas_optics_lw = gas_optics_lw[LW_ORDER]
# convert str vars to char8
for vars in ["xsec_species_names", "fax_species_names", "mtckd_species_names"]:
    gas_optics_lw[vars] = gas_optics_lw[vars].astype("S32")
    gas_optics_lw[vars].encoding["dtype"] = "S1"


gas_optics_lw.to_netcdf("../../ddq-data/gas_optics_lw.nc")


# =============================================================================
# %% Shortwave
# =============================================================================

data_sw = xr.open_datatree("../data/ff/test_3_sw.nc").copy()

# ---- Hinge rational ---------------------------------------------------------

lines = data_sw["Hinge_Rational"].to_dataset().rename(LINE_RENAME)

lines["fax_p_nterms"] = ("fax_p_nterms", range(lines.sizes["fax_p_nterms"]))

lines["fax_a"] = (
    lines["temperature_coeffs"]
    .isel(t_order=[0, 1, 2])
    .rename({"t_order": "fax_t_order"})
    .assign_coords(fax_t_order=[0, 1, 2])
)

ones = xr.ones_like(lines["temperature_coeffs"].isel(t_order=0, drop=True))
ones = ones.expand_dims(fax_t_order=[0])

rest = (
    lines["temperature_coeffs"]
    .isel(t_order=[3, 4])
    .rename({"t_order": "fax_t_order"})
    .assign_coords(fax_t_order=[1, 2])
)

lines["fax_b"] = xr.concat([ones, rest], dim="fax_t_order")

lines = lines.drop_vars(["temperature_coeffs", "t_order", "fax_vmr0"])
lines["fax_species_names"] = lines["fax_species_names"].str.lower()

# ---- MTCKD ------------------------------------------------------------------

cont = data_sw["both_continuum_MT_CKD_4_3"].to_dataset().rename(CONT_RENAME)
cont["mtckd_species_names"] = cont["mtckd_species_names"].str.lower()

# ---- XFIT -------------------------------------------------------------------

xsec = (
    xr.concat(
        [data_sw["XFIT"].to_dataset()[name] for name in TERM_NAMES],
        dim="xsec_nterms",
    )
    .assign_coords(xsec_nterms=range(4))
    .to_dataset()
    .rename(
        {
            "species": "xsec_species_names",
            "p00": "xsec_p",
        }
    )
)
xsec["xsec_species_names"] = xsec["xsec_species_names"].str.lower()

# ---- Merge ------------------------------------------------------------------

gas_optics_sw = xr.merge([lines, cont, xsec])

gas_optics_sw["frequency"] = hz_to_kayser(gas_optics_sw["frequency"])
gas_optics_sw = gas_optics_sw.rename({"frequency": "nu"})

gas_optics_sw["weights"] = (
    "nu",
    hz_to_kayser(data_sw["DDQ"]["weights_hz"].values),
)

gas_optics_sw["solar_spectral_irradiance"] = (
    "nu",
    data_sw["DDQ"]["spectral_solar_irradiance"].values,  # * LIGHT_SPEED * CM_TO_M,
)
gas_optics_sw["rayleigh_xsec"] = (
    "nu",
    data_sw["DDQ"]["xsec_rayleigh"].values,
)
gas_optics_sw = clear_all_attrs(gas_optics_sw)
gas_optics_sw = gas_optics_sw.transpose(*TRANSPOSE_ORDER)


gas_optics_sw = gas_optics_sw.assign_coords(
    fax_nspecies=("fax_species_names", range(gas_optics_sw.sizes["fax_species_names"])),
    xsec_nspecies=(
        "xsec_species_names",
        range(gas_optics_sw.sizes["xsec_species_names"]),
    ),
    mtckd_nspecies=(
        "mtckd_species_names",
        range(gas_optics_sw.sizes["mtckd_species_names"]),
    ),
)

gas_optics_sw = (
    gas_optics_sw.swap_dims({"fax_species_names": "fax_nspecies"})
    .swap_dims({"xsec_species_names": "xsec_nspecies"})
    .swap_dims({"mtckd_species_names": "mtckd_nspecies"})
    .reset_coords(["fax_species_names", "xsec_species_names", "mtckd_species_names"])
)
gas_optics_sw = gas_optics_sw[SW_ORDER]
for vars in ["xsec_species_names", "fax_species_names", "mtckd_species_names"]:
    gas_optics_sw[vars] = gas_optics_sw[vars].astype("S32")
    gas_optics_sw[vars].encoding["dtype"] = "S1"

gas_optics_sw.to_netcdf("../../ddq-data/gas_optics_sw.nc")

# %%
# %%
"""
netcdf gas_optics_sw {
dimensions:
	nu = 64 ;
	fax_nspecies = 7 ;
	fax_p_nterms = 4 ;
	fax_t_order = 3 ;
	xsec_nspecies = 3 ;
	xsec_nterms = 4 ;
	mtckd_nspecies = 1 ;
variables:
	double nu(nu) 
	double weights(nu) ;
	double solar_spectral_irradiance(nu) ;

	int64 fax_nspecies(fax_nspecies) ;
	char fax_species_names(fax_nspecies) ;
	double fax_p0(fax_nspecies) ;
	double fax_T0(fax_nspecies) ;
	double fax_S(fax_nspecies) ;
	double fax_sigma0(fax_nspecies, nu) ;
	int64 fax_p_nterms(fax_p_nterms) ;
	double fax_c(fax_p_nterms, fax_nspecies, nu) ;
	int64 fax_t_order(fax_t_order) ;
	double fax_a(fax_t_order, fax_nspecies, nu) ;
	double fax_b(fax_t_order, fax_nspecies, nu) ;

	int64 xsec_nspecies(xsec_nspecies) ;
	char xsec_species_names(xsec_nspecies) ;
	int64 xsec_nterms(xsec_nterms) ;
	double xsec_p(xsec_nterms, xsec_nspecies, nu) ;
    
	int64 mtckd_nspecies(mtckd_nspecies) ;
	char mtckd_species_names(mtckd_nspecies) ;
	double mtckd_p0(mtckd_nspecies) ;
	double mtckd_T0(mtckd_nspecies) ;
	double mtckd_cself(mtckd_nspecies, nu) ;
	double mtckd_n(mtckd_nspecies, nu) ;
	double mtckd_cfrgn(mtckd_nspecies, nu) ;
	double rayleigh_xsec(nu) ;
}

"""

# %%
