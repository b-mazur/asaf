from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from asaf.constants import _ATM_TO_PA, _MOLAR_GAS_CONSTANT, atomic_mass
from gemmi import cif

from itertools import product

if TYPE_CHECKING:
    from typing import List, Optional

logger = logging.getLogger(__name__) 

class Framework(object):
    """
    Represents a molecular structure read from CIF.
    """
    def __init__(self, cif_file: Path, remove_site_labels: bool = False):
        self._cif_file = Path(cif_file)
        self._dataframe = None
        self._cell_lengths = None
        self._cell_angles = None
        self._framework_mol_mass = None
        self._molecules_uc__mol_kg = None
        self._molecules_uc__cm3_g = None
        self._force_field = None
        self._site_types = None

        self.read_cif(cif_file=cif_file, remove_site_labels=remove_site_labels)
        self.calculate_conversion_factors()

    def read_cif(self, cif_file: Path, remove_site_labels: bool = False) -> None:
        """
        Read the CIF file and populate self._dataframe.
        """
        logger.info("Reading CIF file: %s", cif_file)
        try:
            cif_data = cif.read(str(cif_file))
        except Exception as e:
            raise ValueError(f"Unable to read CIF from {cif_file}: {e}")

        block = cif_data.sole_block()
        # Check space group
        try:
            sg = block.find_value("_symmetry_space_group_name_H-M")
            sg_clean = ''.join(ch for ch in sg if ch.isalnum())
            if sg_clean.lower() != "p1":
                raise ValueError(f"CIF in {sg}, only P1 symmetry space is supported.")
            else:
                atom_site_labels = list(block.find_loop("_atom_site_label"))
                atom_site_types = list(block.find_loop("_atom_site_type_symbol"))
                atom_site_fract_x = list(block.find_loop("_atom_site_fract_x"))
                atom_site_fract_y = list(block.find_loop("_atom_site_fract_y"))
                atom_site_fract_z = list(block.find_loop("_atom_site_fract_z"))
                atom_site_charges = list(block.find_loop("_atom_site_charge"))
        except Exception as e:
            raise ValueError(f"Error parsing CIF tags: {e}")

        df = pd.DataFrame(
            {
                "_atom_site_label": atom_site_labels,
                "_atom_site_type_symbol": atom_site_types,
                "_atom_site_fract_x": pd.to_numeric(atom_site_fract_x),
                "_atom_site_fract_y": pd.to_numeric(atom_site_fract_y),
                "_atom_site_fract_z": pd.to_numeric(atom_site_fract_z),
                "_atom_site_charge": pd.to_numeric(atom_site_charges),
            }
        )
        if remove_site_labels:
            df["_atom_site_label"] = df["_atom_site_label"].str.rstrip("_0123456789")
        self._dataframe = df

        cell_lengths = (
            float(block.find_value("_cell_length_a")),
            float(block.find_value("_cell_length_b")),
            float(block.find_value("_cell_length_c"))
        )

        cell_angles = (
            float(block.find_value("_cell_angle_alpha")),
            float(block.find_value("_cell_angle_beta")),
            float(block.find_value("_cell_angle_gamma"))
        )

        self._cell_lengths = cell_lengths
        self._cell_angles = cell_angles

    def calculate_framework_mass(self):
        """Calculates the molar mass of the framework.
        Units: g / mol / unit cell"""

        if self._framework_mol_mass is None:
            masses = self._dataframe["_atom_site_type_symbol"].map(atomic_mass)
            if masses.isnull().any():
                missing = self._dataframe["_atom_site_type_symbol"][
                    masses.isnull()
                ].unique()
                raise KeyError(f"No atomic mass found for element(s): {missing}")
            self._framework_mol_mass = masses.sum()
        return self._framework_mol_mass

    def calculate_conversion_factors(self):
        """Calculates the conversion factors for the isotherm recalculation."""

        mass = self.calculate_framework_mass()
        # molecules / unit cell -> mol / kg
        self._molecules_uc__mol_kg = 1000 / mass
        # molecules / unit cell -> cm3 / g
        self._molecules_uc__cm3_g = (
            1.0e6 * (_MOLAR_GAS_CONSTANT * 273.15 / _ATM_TO_PA) / mass
        )

    def site_labels(self, as_list: bool = False) -> pd.Series | List[str]:
        if as_list:
            return self._dataframe["_atom_site_label"].to_list()
        else:
            return self._dataframe["_atom_site_label"]

    def site_types(self, as_list: bool = False) -> pd.Series | List[str]:
        if as_list:
            return self._dataframe["_atom_site_type"].to_list()
        else:
            return self._dataframe["_atom_site_type_symbol"]

    def check_net_charge(self, unit_cells: tuple[int, int, int]) -> float:
        """
        Returns the total net charge of the replicated system (all UC).
        Prints a warning if |net_charge| > 1e-5 e.

        Args:
            unit_cells: how many times to replicate in (x, y, z).
        Returns:
            float: net charge in the full system (units of e).
        """
        total_uc_charge = self._dataframe["_atom_site_charge"].sum()
        system_charge = total_uc_charge * int(np.prod(unit_cells))
        if abs(system_charge) > 1e-5:
            logger.warning("System has net charge = %.5e e. Consider adjusting charges.", system_charge)
        return system_charge

    def reduce_net_charge(self):
        """
        Remove any net charge by proportionally subtracting from each site:
        q_i_new = q_i_old - (sum_j q_j) * (|q_i_old| / sum_k |q_k_old|)

        After this, the total charge across all sites is zero.
        """
        total = self._dataframe["_atom_site_charge"].sum()
        if abs(total) < 1e-12:
            logger.info("Net charge is already ~0 (%.3e). No adjustment needed.", total)
            return

        abs_sum = self._dataframe["_atom_site_charge"].abs().sum()
        if abs_sum == 0:
            logger.error("All atomic charges are zero, cannot reduce net charge.")
            return

        correction = total * self._dataframe["_atom_site_charge"] / abs_sum
        self._dataframe["_atom_site_charge"] = self._dataframe["_atom_site_charge"] - correction
        resid = self._dataframe["_atom_site_charge"].sum()
        self._dataframe["_atom_site_charge"] -= resid / len(self._dataframe["_atom_site_charge"])

        logger.info("Adjusted charges to remove net charge (%.3e).", self._dataframe["_atom_site_charge"].sum())

    def create_system(self, unit_cells: tuple[int, int, int]):
        a0, b0, c0 = self._cell_lengths
        alpha, beta, gamma = np.radians(self._cell_angles)
        nx, ny, nz = unit_cells

        # supercell lengths
        a = a0 * nx
        b = b0 * ny
        c = c0 * nz

        # Transformation formula to convert from fractional space to cartesian
        # source: http://dx.doi.org/10.1080/08927022.2013.819102
        #
        #     ( a   b cos(gamma)   c cos(beta)               )
        # h = ( 0   b sin(gamma)   c z                       )
        #     ( 0   0              c sqrt(1-cos^2(beta)-z^2) )
        #
        # z = (cos(alpha) - cos(gamma) cos(beta)) / sin(gamma)
        #
        # s - fractional coordinates, r - Cartesian coordinates
        # s = h^-1 r,  r = hs

        z = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)

        h = np.array(
            [
                [a, b * np.cos(gamma), c * np.cos(beta)],
                [0.0, b * np.sin(gamma), c * z],
                [0.0, 0.0, c * np.sqrt(1 - np.cos(beta) ** 2 - z**2)],
            ]
        )

        fractional_coordinates = self._dataframe[
            ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
        ].to_numpy()  # (n_sites, 3)
        labels = self.site_labels()

        shifts = np.array(
            list(product(range(nx), range(ny), range(nz)))
        )  # shape (N_cells,3)

        fractional_coordinates_supercell = (
            fractional_coordinates[None, :, :] + shifts[:, None, :]
        ) / np.array([nx, ny, nz])
        cartesian_coordinates_supercell = (
            fractional_coordinates_supercell.reshape(-1, 3) @ h.T
        )

        a_vec = h[:, 0]
        b_vec = h[:, 1]
        c_vec = h[:, 2]

        center = 0.5 * (a_vec + b_vec + c_vec)
        cartesian_coordinates_supercell = cartesian_coordinates_supercell - center

        n_cells = shifts.shape[0]

        system = pd.DataFrame(
            {
                "_atom_site_label": np.tile(labels, n_cells),
                "_atom_site_cartesian_x": cartesian_coordinates_supercell[:, 0],
                "_atom_site_cartesian_y": cartesian_coordinates_supercell[:, 1],
                "_atom_site_cartesian_z": cartesian_coordinates_supercell[:, 2],
            }
        )

        # Calculating FEASST style box properties for system configuration
        lx = a
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        ly = np.sqrt(b**2 - xy**2)
        yz = ((b * c * np.cos(alpha)) - (xy * xz)) / ly
        lz = np.sqrt(c**2 - xz**2 - yz**2)

        def _reduce(skew, box_size):
            """Bring skew into ±box_size/2 by subtracting or adding box length as needed."""
            while 2 * abs(skew) > box_size:
                skew = skew - np.sign(skew) * box_size
            return skew

        if abs(xy) * 2 > lx:
            old = xy
            xy = _reduce(xy, lx)
            logger.warning(
                "Triclinic box skew xy is too large. Reduced from %.4f to %.4f", old, xy
            )

        if abs(xz) * 2 > lx:
            old = xz
            xz = _reduce(xz, lx)
            logger.warning(
                "Triclinic box skew xz is too large. Reduced from %.4f to %.4f", old, xz
            )

        if abs(yz) * 2 > ly:
            old = yz
            yz = _reduce(yz, ly)
            logger.warning(
                "Triclinic box skew yz is too large. Reduced from %.4f to %.4f", old, yz
            )

        box = [lx, ly, lz, xy, xz, yz]
        box = tuple([round(v, 14) for v in box])

        vectors = (a_vec, b_vec, c_vec)

        return system, box, vectors

    # def write_metadata(self, metadata_file_name, box, unit_cells, cutoff, alpha, kmax, cell_vectors):
    def write_metadata(self, metadata_file_name, box, unit_cells, cutoff, cell_vectors):
        """Write metadata to a separate file."""

        metadata = {
            "box_size": box[:3],
            "tilt_factors": box[3:],
            "lattice": list(list(vec) for vec in cell_vectors),
            "unit_cells": list(unit_cells),
            "cell_lengths": list(self._cell_lengths),
            "cell_angles": list(self._cell_angles),
            "cutoff": cutoff,
            # "alpha": alpha,
            # "kmax": kmax,
            "molecules/unitcell_to_cm3stp/g": self._molecules_uc__cm3_g,
            "molecules/unitcell_to_mol/kg": self._molecules_uc__mol_kg,
        }

        logger.info("Writing metadata to %s.metadata.json", metadata_file_name)
        with open(f"{metadata_file_name}.metadata.json", "w") as metadata_f_out:
            json.dump(metadata, metadata_f_out, indent=4)

        return metadata

    def get_site_types(self):
        self._site_types = (
            self._dataframe.groupby(by="_atom_site_label").first().reset_index()
        )

    def set_force_field(
        self,
        sigmas: Optional[dict] = None,
        sigma_by: str = "_atom_site_type_symbol",
        epsilons: Optional[dict] = None,
        epsilon_by: str = "_atom_site_type_symbol",
        charges: Optional[dict] = None,
        charge_by: str = "_atom_site_label",
    ):
        self.get_site_types()

        if self._force_field is not None and not self._force_field.empty:
            ff = self._force_field.copy()
        else:
            ff = self._site_types[
                ["_atom_site_label", "_atom_site_type_symbol", "_atom_site_charge"]
            ].copy()

        ff["site"] = ff.index

        if sigmas is None:
            if "sigma" not in ff:
                ff["sigma"] = np.nan
        else:
            ff["sigma"] = ff[sigma_by].map(sigmas)

        if epsilons is None:
            if "epsilon" not in ff:
                ff["epsilon"] = np.nan
        else:
            ff["epsilon"] = ff[epsilon_by].map(epsilons)

        if charges is None:
            if "_atom_site_charge" not in ff:
                ff["_atom_site_charge"] = np.nan
        else:
            ff["_atom_site_charge"] = ff[charge_by].map(charges)

        self._force_field = ff

    def average_charges(self, tolerance:float = 0.1):
        from pymatgen.core.structure import Structure
        from pymatgen.core.lattice import Lattice
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
        from collections import defaultdict

        a, b, c = self._cell_lengths
        alpha, beta, gamma = self._cell_angles

        lattice = Lattice.from_parameters(
            a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
        )

        structure = Structure(
            lattice=lattice,
            species=self._dataframe._atom_site_type_symbol.to_list(),
            coords=self._dataframe[
                ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
            ].to_numpy(),
            labels=self._dataframe._atom_site_label.to_list(),
            site_properties={"charge": self._dataframe._atom_site_charge.to_list()},
        )

        n_sites = len(structure)
        radii = []
        for site in structure:
            r = CovalentRadius.radius.get(site.specie.symbol, None)
            if r is None:
                raise ValueError(f"No covalent radius for element {site.specie}")
            radii.append(r)
        max_rad = max(radii)

        max_cutoff = 2 * max_rad + tolerance
        all_nbrs = structure.get_all_neighbors(max_cutoff, include_index=True)

        first_shell = {}
        for i, nbr_list in enumerate(all_nbrs):
            bonded = []
            for nb in nbr_list:
                dist = nb[1]
                j = int(nb[2])
                # bond if within sum-of-radii + tolerance
                if dist <= (radii[i] + radii[j] + tolerance):
                    bonded.append(j)
            first_shell[i] = sorted(set(bonded))

        second_shell = {}
        for i in range(n_sites):
            sec = set()
            for j in first_shell[i]:
                sec.update(first_shell[j])
            sec.discard(i)
            sec.difference_update(first_shell[i])
            second_shell[i] = sorted(sec)

        groups = defaultdict(list)
        for i in range(n_sites):
            el = structure[i].specie.symbol
            shell1 = tuple(sorted(structure[j].specie.symbol for j in first_shell[i]))
            shell2 = tuple(sorted(structure[j].specie.symbol for j in second_shell[i]))
            key = (el, shell1, shell2)
            groups[key].append(i)

        logger.info("Found %i groups.", len(groups))







    def write_xyz_file(
        self,
        file_name: str,
        system: pd.DataFrame,
        vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Writes system in extxyz file format."""

        n_sites = system.shape[0]
        flat_vectors = np.concatenate(vectors)
        vectors_str = " ".join(f"{x:.10f}" for x in flat_vectors)

        with open(f"{file_name}.xyz", "w") as xyz_file:
            print(n_sites, file=xyz_file)
            print(
                f"Lattice=\"{vectors_str}\" Properties=species:S:1:pos:R:3", file=xyz_file
            )
            print(system.to_string(header=False, index=False), file=xyz_file)

    def dl_poly_ewald(self, cutoff: float, box: tuple, tolerance: float = 0.00001):
        """Parameters from the DL_POLY Algorithm
        https://doi.org/10.1080/002689798167881
        thanks to Daniel W. Siderius
        """
        # DL_POLY Algorithm
        eps = min(tolerance, 0.5)
        xi = np.sqrt(np.abs(np.log(eps * cutoff)))
        alpha = np.sqrt(np.abs(np.log(eps * cutoff * xi))) / cutoff
        chi = np.sqrt(-np.log(eps * cutoff * ((2.0 * xi * alpha) ** 2)))
        kmax = [int(0.25 + box[i] * alpha * chi / np.pi) for i in range(3)]

        return alpha, kmax

    def write_fstprt(
        self,
        unit_cells: tuple[int, int, int] = (1, 1, 1),
        file_name: Optional = None,
        cutoff: float = 12.8,
        return_metadata: bool = False,
        ewald_tolerance: float = 0.00001
    ) -> None | dict:
        """Writes molecule file with framework for FEASST simulation software."""

        if len(unit_cells) != 3 or not all(isinstance(n, int) for n in unit_cells):
            raise ValueError("`unit_cells` must be three positive integers")

        system, box, vectors = self.create_system(unit_cells)
        net_charge = self.check_net_charge(unit_cells)
        logger.info("Net charge is %e", net_charge)
        # alpha, kmax = self.dl_poly_ewald(
        #     cutoff=cutoff, box=box, tolerance=ewald_tolerance
        # )

        n_sites = system.shape[0]

        if file_name is None:
            file_name = self._cif_file.with_suffix("")

        metadata = self.write_metadata(
            metadata_file_name=file_name,
            box=box,
            unit_cells=unit_cells,
            cutoff=cutoff,
            # alpha=alpha,
            # kmax=kmax,
            cell_vectors=vectors,
        )

        self.write_xyz_file(file_name, system, vectors)

        self.set_force_field()
        sites_to_num = pd.Series(
            self._force_field.site.values, index=self._force_field._atom_site_label
        ).to_dict()
        system["_atom_site_label"] = (
            system["_atom_site_label"].map(sites_to_num).astype(int)
        )

        file = f"""#

{n_sites} sites

{self._force_field.shape[0]} site types

Units

length Angstrom
energy kJ/mol
charge elementary

Site Labels

{self._force_field["_atom_site_label"].to_string()}

Site Properties

"""
        for i, row in self._force_field.iterrows():
            line = (
                    f"{i:<2} "
                    + f"sigma {row['sigma']:<7.5f}  "
                    + f"epsilon {row['epsilon']:<10.8f}  "
                    + f"cutoff {cutoff:<3.1f}  "
                    + f"charge {row['_atom_site_charge']:>18.15f}\n"
            )
            file += line

        file += "\nSites\n\n"

        file += system.to_string(header=False)

        file += "\n"

        with open(f"{file_name}.fstprt", "w") as fstprt_file:
            print(file, file=fstprt_file)

        if return_metadata:
            return metadata
        else:
            return None

