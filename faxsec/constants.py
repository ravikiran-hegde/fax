"""Numerical and physical constants."""

BOLTZMANN = 1.380649e-23
LIGHT_SPEED = 2.99792458e8
PLANCK = 6.62607015e-34
CP_AIR = 1004.0
GRAVITY = 9.80665
MEAN_MOLAR_MASS_AIR = 28.9647e-3
MEAN_MOLAR_MASS_H2O = 18.01528e-3
AVOGADRO = 6.02214076e23
MEAN_MASS_AIR = 4.80970159e-26  # MEAN_MOLAR_MASS_AIR / AVOGADRO
RADCN2 = 1.4387752e-2  # PLANCK * LIGHT_SPEED / BOLTZMANN

S_TO_DAY = 86400.0
CM_TO_M = 100.0
KAYSER_TO_HZ = 29979245800.0  # CM_TO_M * LIGHT_SPEED
HZ_TO_KAYSER = 3.33564095198152e-11  # 1.0 / KAYSER_TO_HZ

REF_PRESSURE = 1.0
REF_TEMPERATURE = 150.0
REF_VMR = 1e-9

EPS = 1e-200

SELF_SCALING = (
    {  # unweighted median ratio of self-broadened to air-broadened half-widths
        "H2O": 4.078,
        "CO2": 0.282,
        "O3": 0.247,
        "CH4": 0.311,
        "N2O": 0.254,
        "CO": 0.067,
        "O2": 0.004,
        "N2": 0.006,
    }
)

DEFAULT_VMR = {
    "N2": 0.7808,
    "O2": 0.2095,
    "CO2": 4.2e-4,
    "H2O": 0.001,
    "CH4": 1.9e-6,
    "N2O": 3.3e-7,
    "CFC11": 2.5e-10,
    "CFC12": 5.0e-10,
}
