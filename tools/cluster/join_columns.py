"""Join the per-column flux files of rte_array.sh into one file per backend and case.

    python tools/cluster/join_columns.py arts-highres rfmip
"""

import re
import sys

import xarray as xr

from artsfax.constants import DATA_DIR


def main(backend: str, case: str) -> None:
    folder = DATA_DIR / "rte_examples"
    # excludes thinned (_s<stride>) test runs
    pattern = re.compile(rf"fluxes_{re.escape(backend)}_{case}_v(\d+)_c(\d+)-(\d+)\.nc")

    by_variant = {}
    for path in sorted(folder.glob(f"fluxes_{backend}_{case}_v*_c*.nc")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        variant, start, stop = map(int, match.groups())
        ds = xr.open_dataset(path).drop_dims("frequency", errors="ignore")
        ds = ds.drop_vars("variant", errors="ignore")
        ds = ds.expand_dims("col") if "col" not in ds.dims else ds
        by_variant.setdefault(variant, []).append(
            ds.assign_coords(col=range(start, stop))
        )
    if not by_variant:
        raise SystemExit(f"no fluxes_{backend}_{case}_v*_c*.nc in {folder}")

    joined = xr.concat(
        [xr.concat(parts, dim="col").sortby("col") for _, parts in sorted(by_variant.items())],
        dim=xr.DataArray(sorted(by_variant), dims="variant", name="variant"),
        join="outer",
    ).transpose("variant", "col", "level")
    for variant in joined["variant"].values:
        complete = joined.sel(variant=variant)["lw_flux_up"].notnull().all("level")
        print(f"variant {variant}: {int(complete.sum())} columns")

    out = folder / f"fluxes_{backend}_{case}.nc"
    joined.to_netcdf(out)
    print(f"wrote {out}: {dict(joined.sizes)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
