"""Common constants and parameters for molecular simulations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List

logger = logging.getLogger(__name__)


_BOLTZMANN_CONSTANT = 1.380649e-23  # J K^-1
_AVOGADRO_CONSTANT = 6.02214076e23  # mol^-1
_MOLAR_GAS_CONSTANT = _BOLTZMANN_CONSTANT * _AVOGADRO_CONSTANT  # J K^-1 mol^-1
_ATM_TO_PA = 101325.0
_KCAL_TO_KJ = 4.184

atomic_mass = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815384,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955907,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938043,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.905838,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Tc": 97,
    "Ru": 101.07,
    "Rh": 102.90549,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Pm": 145,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.925354,
    "Dy": 162.500,
    "Ho": 164.930329,
    "Er": 167.259,
    "Tm": 168.934219,
    "Yb": 173.045,
    "Lu": 174.9668,
    "Hf": 178.486,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966570,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98040,
    "Po": 209,
    "At": 210,
    "Rn": 222,
    "Fr": 223,
    "Ra": 226,
    "Ac": 227,
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
    "Np": 237,
    "Pu": 244,
    "Am": 243,
    "Cm": 247,
    "Bk": 247,
    "Cf": 251,
    "Es": 252,
    "Fm": 257,
    "Md": 258,
    "No": 259,
    "Lr": 262,
    "Rf": 267,
    "Db": 270,
    "Sg": 269,
    "Bh": 270,
    "Hs": 270,
    "Mt": 278,
    "Ds": 281,
    "Rg": 281,
    "Cn": 285,
    "Nh": 286,
    "Fl": 289,
    "Mc": 289,
    "Lv": 293,
    "Ts": 293,
    "Og": 294,
}

# source: Cordero, B. et al. (2008). Covalent radii revisited. Dalton Transactions, (21), 2832-2838.
# DOI: 10.1039/b801115j
covalent_radii = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.40,
    "At": 1.50,
    "Rn": 1.50,
    "Fr": 2.60,
    "Ra": 2.21,
    "Ac": 2.15,
    "Th": 2.06,
    "Pa": 2.00,
    "U": 1.96,
    "Np": 1.90,
    "Pu": 1.87,
    "Am": 1.80,
    "Cm": 1.69,
}

class ForceFieldParameters:
    """
    Container for most common force fields in a conventional 12-6 LJ potential.

    Currently implemented:
    - Dreiding
    - UFF
    """

    # DREIDING parameters (r0 in Å, D in kcal/mol)
    # source: https://doi.org/10.1021/j100389a010, TABLE II
    _DREIDING = {
        "H": {"r0": 3.195, "D": 0.0152},
        "H_HB": {"r0": 3.195, "D": 0.0001},
        "B": {"r0": 4.02, "D": 0.095},
        "C": {"r0": 3.8983, "D": 0.0951},
        "N": {"r0": 3.6621, "D": 0.0774},
        "O": {"r0": 3.4046, "D": 0.0957},
        "F": {"r0": 3.472, "D": 0.0725},
        "Al": {"r0": 4.39, "D": 0.31},
        "Si": {"r0": 4.27, "D": 0.31},
        "P": {"r0": 4.15, "D": 0.32},
        "S": {"r0": 4.03, "D": 0.344},
        "Cl": {"r0": 3.9503, "D": 0.2833},
        "Ga": {"r0": 4.39, "D": 0.4},
        "Ge": {"r0": 4.27, "D": 0.4},
        "As": {"r0": 4.15, "D": 0.41},
        "Se": {"r0": 4.03, "D": 0.43},
        "Br": {"r0": 3.95, "D": 0.37},
        "In": {"r0": 4.59, "D": 0.55},
        "Sn": {"r0": 4.47, "D": 0.55},
        "Sb": {"r0": 4.35, "D": 0.55},
        "Te": {"r0": 4.23, "D": 0.57},
        "I": {"r0": 4.15, "D": 0.51},
        "Na": {"r0": 3.144, "D": 0.5},
        "Ca": {"r0": 3.472, "D": 0.05},
        "Fe": {"r0": 4.54, "D": 0.055},
        "Zn": {"r0": 4.54, "D": 0.055},
        "C_R1": {"r0": 4.23, "D": 0.1356},
        "C_34": {"r0": 4.237, "D": 0.3016},
        "C_33": {"r0": 4.1524, "D": 0.25},
        "C_32": {"r0": 4.0677, "D": 0.1984},
        "C_31": {"r0": 3.983, "D": 0.1467},
    }

    # UFF parameters (r0 in Å, D in kcal/mol)
    # source: https://doi.org/10.1021/ja00051a040, Table I
    _UFF = {
        "H": {"r0": 2.886, "D": 0.044},
        "He": {"r0": 2.362, "D": 0.056},
        "Li": {"r0": 2.451, "D": 0.025},
        "Be": {"r0": 2.745, "D": 0.085},
        "B": {"r0": 4.083, "D": 0.180},
        "C": {"r0": 3.851, "D": 0.105},
        "N": {"r0": 3.660, "D": 0.069},
        "O": {"r0": 3.500, "D": 0.060},
        "F": {"r0": 3.364, "D": 0.050},
        "Ne": {"r0": 3.243, "D": 0.042},
        "Na": {"r0": 2.983, "D": 0.030},
        "Mg": {"r0": 3.021, "D": 0.111},
        "Al": {"r0": 4.499, "D": 0.505},
        "Si": {"r0": 4.295, "D": 0.402},
        "P": {"r0": 4.147, "D": 0.305},
        "S": {"r0": 4.035, "D": 0.274},
        "Cl": {"r0": 3.947, "D": 0.227},
        "Ar": {"r0": 3.868, "D": 0.185},
        "K": {"r0": 3.812, "D": 0.035},
        "Ca": {"r0": 3.399, "D": 0.238},
        "Sc": {"r0": 3.295, "D": 0.019},
        "Ti": {"r0": 3.175, "D": 0.017},
        "V": {"r0": 3.144, "D": 0.016},
        "Cr": {"r0": 3.023, "D": 0.015},
        "Mn": {"r0": 2.961, "D": 0.013},
        "Fe": {"r0": 2.912, "D": 0.013},
        "Co": {"r0": 2.872, "D": 0.014},
        "Ni": {"r0": 2.834, "D": 0.015},
        "Cu": {"r0": 3.495, "D": 0.005},
        "Zn": {"r0": 2.763, "D": 0.124},
        "Ga": {"r0": 4.383, "D": 0.415},
        "Ge": {"r0": 4.280, "D": 0.379},
        "As": {"r0": 4.230, "D": 0.309},
        "Se": {"r0": 4.205, "D": 0.291},
        "Br": {"r0": 4.189, "D": 0.251},
        "Kr": {"r0": 4.141, "D": 0.220},
        "Rb": {"r0": 4.114, "D": 0.040},
        "Sr": {"r0": 3.641, "D": 0.235},
        "Y": {"r0": 3.345, "D": 0.072},
        "Zr": {"r0": 3.124, "D": 0.069},
        "Nb": {"r0": 3.165, "D": 0.059},
        "Mo": {"r0": 3.052, "D": 0.056},
        "Tc": {"r0": 2.998, "D": 0.048},
        "Ru": {"r0": 2.963, "D": 0.056},
        "Rh": {"r0": 2.929, "D": 0.053},
        "Pd": {"r0": 2.899, "D": 0.048},
        "Ag": {"r0": 3.148, "D": 0.036},
        "Cd": {"r0": 2.848, "D": 0.228},
        "In": {"r0": 4.463, "D": 0.599},
        "Sn": {"r0": 4.392, "D": 0.567},
        "Sb": {"r0": 4.420, "D": 0.449},
        "Te": {"r0": 4.470, "D": 0.398},
        "I": {"r0": 4.500, "D": 0.339},
        "Xe": {"r0": 4.404, "D": 0.332},
        "Cs": {"r0": 4.517, "D": 0.045},
        "Ba": {"r0": 3.703, "D": 0.364},
        "La": {"r0": 3.522, "D": 0.017},
        "Ce": {"r0": 3.556, "D": 0.013},
        "Pr": {"r0": 3.606, "D": 0.010},
        "Nd": {"r0": 3.575, "D": 0.010},
        "Pm": {"r0": 3.547, "D": 0.009},
        "Sm": {"r0": 3.520, "D": 0.008},
        "Eu": {"r0": 3.493, "D": 0.008},
        "Gd": {"r0": 3.368, "D": 0.009},
        "Tb": {"r0": 3.451, "D": 0.007},
        "Dy": {"r0": 3.428, "D": 0.007},
        "Ho": {"r0": 3.409, "D": 0.007},
        "Er": {"r0": 3.391, "D": 0.007},
        "Tm": {"r0": 3.374, "D": 0.006},
        "Yb": {"r0": 3.355, "D": 0.228},
        "Lu": {"r0": 3.640, "D": 0.041},
        "Hf": {"r0": 3.141, "D": 0.072},
        "Ta": {"r0": 3.170, "D": 0.081},
        "W": {"r0": 3.069, "D": 0.067},
        "Re": {"r0": 2.954, "D": 0.066},
        "Os": {"r0": 3.120, "D": 0.037},
        "Ir": {"r0": 2.840, "D": 0.073},
        "Pt": {"r0": 2.754, "D": 0.080},
        "Au": {"r0": 3.293, "D": 0.039},
        "Hg": {"r0": 2.705, "D": 0.385},
        "Tl": {"r0": 4.347, "D": 0.680},
        "Pb": {"r0": 4.297, "D": 0.663},
        "Bi": {"r0": 4.370, "D": 0.518},
        "Po": {"r0": 4.709, "D": 0.325},
        "At": {"r0": 4.750, "D": 0.284},
        "Rn": {"r0": 4.765, "D": 0.248},
        "Fr": {"r0": 4.900, "D": 0.050},
        "Ra": {"r0": 3.677, "D": 0.404},
        "Ac": {"r0": 3.478, "D": 0.033},
        "Th": {"r0": 3.396, "D": 0.026},
        "Pa": {"r0": 3.424, "D": 0.022},
        "U": {"r0": 3.395, "D": 0.022},
        "Np": {"r0": 3.424, "D": 0.019},
        "Pu": {"r0": 3.424, "D": 0.016},
        "Am": {"r0": 3.381, "D": 0.014},
        "Cm": {"r0": 3.326, "D": 0.013},
        "Bk": {"r0": 3.339, "D": 0.013},
        "Cf": {"r0": 3.313, "D": 0.013},
        "Es": {"r0": 3.299, "D": 0.012},
        "Fm": {"r0": 3.286, "D": 0.012},
        "Md": {"r0": 3.274, "D": 0.011},
        "No": {"r0": 3.248, "D": 0.011},
        "Lw": {"r0": 3.236, "D": 0.011},
        "Lr": {"r0": 3.236, "D": 0.011},  # formerly Lw, now Lr
    }

    _FORCE_FIELDS = {
        "dreiding": _DREIDING,
        "uff": _UFF,
    }

    def __init__(self, atom_types: List[str], name: str, unit: str = "kj/mol"):
        """
        Initialize the ForceFieldParameters object.

        Args:
            atom_types (List[str]): List of atom types or element symbols
            name (str): Name of the force field ("dreiding" or "uff"), case-insensitive
            unit (str): Unit for the energy parameter, default is "kj/mol"
        """
        self.name = name.lower()
        self.unit = unit.lower()
        self.parameters = {}

        if self.name not in self._FORCE_FIELDS:
            raise ValueError(
                f"Force field '{self.name}' is not supported. Supported force fields: {list(self._FORCE_FIELDS.keys())}"
            )

        for atom in atom_types:
            self._add_atom(atom)

    @staticmethod
    def _convert_r0_to_sigma(r0: float) -> float:
        """
        Convert r0 to sigma for standard 12-6 LJ potential.

        Arguments
        ---------
        r0 : float
            r0 value in Å

        Returns
        -------
        float
            sigma value in Å
        """
        return r0 / (2 ** (1 / 6))

    def _convert_d_to_epsilon(self, d: float) -> float:
        """
        Convert D to epsilon for standard 12-6 LJ potential.

        Arguments
            d : float
        D value in kcal/mol

        Returns
        -------
        float
            epsilon value in selected energy unit
        """
        if self.unit == "kj/mol":
            return d * _KCAL_TO_KJ
        elif self.unit == "kcal/mol":
            return d
        elif self.unit == "K":
            return (
                d * _KCAL_TO_KJ / _MOLAR_GAS_CONSTANT * 1e3
            )  # kcal mol^-1 * kJ kcal^-1 * J^-1 K mol * 1e3 = K
        else:
            raise ValueError(
                f"Unsupported energy unit '{self.unit}'. Supported units: 'kj/mol', 'kcal/mol'."
            )

    def _add_atom(self, atom: str):
        """
        Add an atom type to the force field parameters.

        Arguments
        ---------
        atom : str
            Atom type or element symbol
        """
        try:
            if self.name == "uff" and atom in self._UFF:
                atom_type = atom
            elif self.name == "dreiding" and atom in self._DREIDING:
                atom_type = atom
            else:
                logging.warning("Unknown atom type '%s'.", atom)
                atom_type = None

            if atom_type:
                parameters = self._FORCE_FIELDS[self.name][atom_type]

                r0 = parameters["r0"]
                d = parameters["D"]

                sigma = self._convert_r0_to_sigma(r0)
                epsilon = self._convert_d_to_epsilon(d)

                self.parameters[atom] = {
                    "sigma": sigma,
                    "sigma_unit": "Å",
                    "epsilon": epsilon,
                    "epsilon_unit": self.unit,
                    "source": f"{self.name.upper()}: {atom_type}",
                }
            else:
                self.parameters[atom] = {
                    "sigma": None,
                    "epsilon": None,
                }

        except KeyError:
            logging.warning("Could not add atom type '%s'.", atom)
            self.parameters[atom] = {
                "sigma": None,
                "epsilon": None,
            }

    @classmethod
    def combine(cls, force_field_dict: Dict[str, List[str]], unit: str = "kj/mol"):
        """
        Combine multiple force fields into a single ForceFieldParameters object.

        Arguments
        ---------
        force_field_dict : (Dict[str, List[str]])
            Dictionary with force field names as keys and lists of atom types as values
        unit : str
            Unit for the energy parameter, default is "kj/mol"
        """
        combined = cls(atom_types=[], name="dreiding", unit=unit)

        for ff_name, atoms in force_field_dict.items():
            temp_ff = cls(atom_types=atoms, name=ff_name, unit=unit)
            combined.parameters.update(temp_ff.parameters)

        return combined

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Return the force field parameters as a dictionary.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary with atom types as keys and their parameters as values
        """
        return self.parameters
