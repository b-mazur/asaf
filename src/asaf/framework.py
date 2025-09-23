"""Framework class for handling molecular periodic structures and generating simulation files."""

from __future__ import annotations

import json
import logging
import math
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gemmi import cif

from asaf.constants import _ATM_TO_PA, _MOLAR_GAS_CONSTANT, atomic_mass, covalent_radii

if TYPE_CHECKING:
    from typing import Dict, List, Optional

    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class Framework(object):
    """Represents a molecular periodic structure.

    The basic components are the sequence of sites and the crystal lattice
    parameters. Additionally, it may store framework-specific and site-specific properties.
    """

    def __init__(
        self,
        lattice: List[float, float, float, float, float, float] | ArrayLike,
        sites: List[str | int],
        coordinates: ArrayLike,
        lattice_as_matrix: bool = False,
        site_types: Optional[List[str]] = None,
        charges: Optional[List[float]] = None,
    ):
        """Initialize the Framework object from lattice parameters, site labels, and coordinates.

        Args:
            lattice : list
                Lattice parameters as a list of six floats representing the lengths (a, b, c) and
                angles (alpha, beta, gamma) of the unit cell or as a 3x3 matrix. If matrix is provided,
                `lattice_as_matrix` should be set to True
            sites : list
                Site labels or indices corresponding to the atomic numbers of the sites
            coordinates : array-like
                2D array of shape (n_sites, 3) containing the fractional coordinates of the sites in the unit cell
            lattice_as_matrix : bool
                If True, `lattice` is treated as a 3x3 matrix representing the unit cell vectors.
                If False, it is treated as a list of six floats representing the lattice parameters
                (a, b, c, alpha, beta, gamma).
        """
        if lattice_as_matrix:
            if lattice.shape != (3, 3):
                raise ValueError(
                    "If `lattice_as_matrix` is True, `lattice` must be a 3x3 matrix."
                )
            self._lattice = lattice
        else:
            if len(lattice) != 6:
                raise ValueError(
                    "If `lattice_as_matrix` is False, `lattice` must be a list of six floats."
                )
            self._cell_lengths = tuple(lattice[:3])  # (a, b, c)
            self._cell_angles = tuple(lattice[3:6])  # (alpha, beta, gamma)
            self._lattice = self.lattice_parameters_to_matrix(
                lattice[0], lattice[1], lattice[2], lattice[3], lattice[4], lattice[5]
            )

        if charges is None:
            charges = np.zeros(len(sites))

        if site_types is None:
            # assuming that site labels are atoms with suffixes like _1, _2, etc.
            site_types = [s.rstrip("_0123456789") for s in sites]

        self._dataframe = pd.DataFrame(
            {
                "site_label": sites,
                "site_type": site_types,
                "fractional_x": coordinates[:, 0],
                "fractional_y": coordinates[:, 1],
                "fractional_z": coordinates[:, 2],
                "site_charge": charges,
            }
        )

        self._framework_mol_mass = None
        self._force_field = {}

    @staticmethod
    def lattice_parameters_to_matrix(a, b, c, alpha, beta, gamma):
        """Convert lattice parameters to a 3x3 matrix representation of the unit cell.

        source: https://dx.doi.org/10.1080/08927022.2013.819102

            ( a   b cos(gamma)   c cos(beta)               )
        h = ( 0   b sin(gamma)   c z                       )
            ( 0   0              c sqrt(1-cos^2(beta)-z^2) )

        z = (cos(alpha) - cos(gamma) cos(beta)) / sin(gamma)

        Here lower triangular form is used, for row -> vector cell convention.

        Args:
            a, b, c (float): lengths of the unit cell edges
            alpha, beta, gamma (float): angles between the edges in degrees

        Returns
        -------
            np.ndarray: 3x3 matrix representing the unit cell vectors, each row is a vector
        """
        alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
        z = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)

        return np.array(
            [
                [a, 0.0, 0.0],
                [b * np.cos(gamma), b * np.sin(gamma), 0.0],
                [c * np.cos(beta), c * z, c * np.sqrt(1 - np.cos(beta) ** 2 - z**2)],
            ]
        )

    def fractional_to_cartesian(self, fractional_coords, lattice=None):
        """Convert fractional coordinates to cartesian coordinates.

        Args:
            fractional_coords: Nx3 array of fractional coordinates
            lattice: Optional 3x3 lattice matrix. If None, uses self._lattice

        Returns
        -------
            Nx3 array of cartesian coordinates
        """
        if lattice is None:
            lattice = self._lattice

        return fractional_coords @ lattice

    @classmethod
    def from_cif(
        cls,
        cif_file: Path,
        remove_site_labels: bool = False,
        partial_charge_header: str = "_atom_site_charge",
    ) -> Framework:
        """Read the CIF file and populate self._dataframe."""
        logger.info("Reading CIF file: %s", cif_file)
        try:
            cif_data = cif.read(str(cif_file))
        except Exception as e:
            raise ValueError(f"Unable to read CIF from {cif_file}: {e}")

        block = cif_data.sole_block()
        # Check space group
        try:
            sg = block.find_value("_symmetry_space_group_name_H-M")
            sg_clean = "".join(ch for ch in sg if ch.isalnum())
            if sg_clean.lower() != "p1":
                raise ValueError(f"CIF in {sg}, only P1 symmetry space is supported.")
            else:
                atom_site_labels = list(block.find_loop("_atom_site_label"))
                atom_site_types = list(block.find_loop("_atom_site_type_symbol"))
                atom_site_fract_x = list(block.find_loop("_atom_site_fract_x"))
                atom_site_fract_y = list(block.find_loop("_atom_site_fract_y"))
                atom_site_fract_z = list(block.find_loop("_atom_site_fract_z"))
                atom_site_charges = list(block.find_loop(partial_charge_header))
        except Exception as e:
            raise ValueError(f"Error parsing CIF tags: {e}")

        coordinates = np.array(
            [atom_site_fract_x, atom_site_fract_y, atom_site_fract_z], dtype=float
        )

        if remove_site_labels:
            atom_site_labels = [s.rstrip("_0123456789") for s in atom_site_labels]

        lattice = [
            float(block.find_value("_cell_length_a")),
            float(block.find_value("_cell_length_b")),
            float(block.find_value("_cell_length_c")),
            float(block.find_value("_cell_angle_alpha")),
            float(block.find_value("_cell_angle_beta")),
            float(block.find_value("_cell_angle_gamma")),
        ]

        return cls(
            lattice=lattice,
            sites=atom_site_labels,
            site_types=atom_site_types,
            coordinates=coordinates.T,  # Transpose to match (n_sites, 3) shape
            lattice_as_matrix=False,
            charges=[float(q) for q in atom_site_charges],
        )

    def calculate_framework_mass(self):
        """Calculate the molar mass of the framework.

        Units: g / mol / unit cell
        """
        if self._framework_mol_mass is None:
            masses = self._dataframe["site_type"].map(atomic_mass)
            if masses.isnull().any():
                missing = self._dataframe["site_type"][masses.isnull()].unique()
                raise KeyError(f"No atomic mass found for element(s): {missing}")
            self._framework_mol_mass = masses.sum()
        return self._framework_mol_mass

    def calculate_conversion_factors(self):
        """Calculate the conversion factors for the isotherm recalculation."""
        mass = self.calculate_framework_mass()
        # molecules / unit cell -> mol / kg
        self._molecules_uc__mol_kg = 1000 / mass
        # molecules / unit cell -> cm3 / g
        self._molecules_uc__cm3_g = (
            1.0e6 * (_MOLAR_GAS_CONSTANT * 273.15 / _ATM_TO_PA) / mass
        )

    def site_labels(self, as_list: bool = False) -> List[str] | pd.Series:
        """Return the site labels as a list or pandas Series."""
        if as_list:
            return self._dataframe["site_label"].to_list()
        else:
            return self._dataframe["site_label"]

    def site_types(self, as_list: bool = False) -> List[str] | pd.Series:
        """Return the site types as a list or pandas Series."""
        if as_list:
            return self._dataframe["site_type"].to_list()
        else:
            return self._dataframe["site_type"]

    def check_net_charge(self, unit_cells: tuple[int, int, int]) -> float:
        """Return the total net charge of the replicated system (all UC).

        Prints a warning if |net_charge| > 1e-5 e.

        Arguments
        ---------
        unit_cells : tuple
            how many times to replicate in (x, y, z).

        Returns
        -------
        float
            net charge in the full system (units of e).
        """
        total_uc_charge = self._dataframe["site_charge"].sum()
        system_charge = total_uc_charge * int(np.prod(unit_cells))
        if abs(system_charge) > 1e-5:
            logger.warning(
                "System has net charge = %.5e e. Consider adjusting charges.",
                system_charge,
            )
        return system_charge

    def reduce_net_charge(self):
        """Remove any net charge by proportionally subtracting from each site.

        The adjustment is done as follows:
        q_i_new = q_i_old - (sum_j q_j) * (|q_i_old| / sum_k |q_k_old|)

        After this, the total charge across all sites is zero.
        #TODO: this averages charges only in dataframe, not in the force field.
        """
        total = self._dataframe["site_charge"].sum()
        if abs(total) < 1e-12:
            logger.info("Net charge is already ~0 (%.3e). No adjustment needed.", total)
            return

        abs_sum = self._dataframe["site_charge"].abs().sum()
        if abs_sum == 0:
            logger.error("All atomic charges are zero, cannot reduce net charge.")
            return

        correction = total * self._dataframe["site_charge"] / abs_sum
        self._dataframe["site_charge"] = self._dataframe["site_charge"] - correction
        resid = self._dataframe["site_charge"].sum()
        self._dataframe["site_charge"] -= resid / len(self._dataframe["site_charge"])

        logger.info(
            "Adjusted charges to remove net charge (%.3e).",
            self._dataframe["site_charge"].sum(),
        )

    @staticmethod
    def _reduce_tilt_factors(box: tuple[float, ...]) -> tuple[float, ...]:
        """Reduce triclinic tilt factors to LAMMPS/FEASST canonical range.

        LAMMPS requires tilt factors to be within specific ranges relative to box lengths:
          xy,xz ∈ (-lx/2, lx/2],  yz ∈ (-ly/2, ly/2]
        https://docs.lammps.org/Howto_triclinic.html#periodicity-and-tilt-factors-for-triclinic-simulation-boxes
        """
        lx, ly, lz, xy, xz, yz = box

        if not (math.isfinite(lx) and math.isfinite(ly)) or lx <= 0 or ly <= 0:
            raise ValueError("Box lengths lx and ly must be positive finite numbers.")

        def remap_tilt(t: float, L: float, tol: float = 1e-12) -> float:
            # Choose integer n so that t' lies in (-L/2, L/2]; +L/2 maps to -L/2
            n = math.floor((t + 0.5 * L) / L)
            t2 = t - n * L
            half = 0.5 * L
            # Post-guard against FP drift
            if t2 > half + tol:
                t2 -= L
            if t2 < -half - tol:
                t2 += L
            # Snap exact endpoints for cleanliness
            if abs(t2 - half) < tol:
                t2 = half
            if abs(t2 + half) < tol:
                t2 = -half
            return float(t2)

        xy_new = remap_tilt(xy, lx)
        xz_new = remap_tilt(xz, lx)
        yz_new = remap_tilt(yz, ly)

        if not (
            np.isclose(xy, xy_new)
            and np.isclose(xz, xz_new)
            and np.isclose(yz, yz_new)
        ):
            parts = []
            if not np.isclose(xy, xy_new):
                parts.append(f"xy: {xy:.6f} → {xy_new:.6f}")
            if not np.isclose(xz, xz_new):
                parts.append(f"xz: {xz:.6f} → {xz_new:.6f}")
            if not np.isclose(yz, yz_new):
                parts.append(f"yz: {yz:.6f} → {yz_new:.6f}")
            if parts:
                logger.info(
                    "Tilt factors reduced: " + ", ".join(parts)
                )

        return lx, ly, lz, xy_new, xz_new, yz_new

    def create_supercell(self, unit_cells=(1, 1, 1), center=True):
        """Create a supercell by replicating the unit cell.

        Args:
            unit_cells: Tuple of (nx, ny, nz) repetitions along each axis
            center: Whether to center the coordinates around origin

        Returns
        -------
            tuple: (
                DataFrame with site_label and cartesian coordinates,
                box dimensions,
                lattice vectors
            )
        """
        nx, ny, nz = unit_cells

        # Create supercell lattice
        supercell_lattice = self._lattice.copy()
        supercell_lattice[0] *= nx
        supercell_lattice[1] *= ny
        supercell_lattice[2] *= nz

        # Get fractional coordinates and labels
        frac_coords = self._dataframe[
            ["fractional_x", "fractional_y", "fractional_z"]
        ].to_numpy()
        labels = self._dataframe["site_label"].to_numpy()

        # Generate all shifts for the supercell
        shifts = np.array(list(product(range(nx), range(ny), range(nz))))

        # Create expanded coordinates and labels
        all_labels = []
        all_frac_coords = []

        for shift in shifts:
            # Apply shift in fractional coordinates
            shifted_coords = frac_coords.copy()
            shifted_coords[:, 0] = (shifted_coords[:, 0] + shift[0]) / nx
            shifted_coords[:, 1] = (shifted_coords[:, 1] + shift[1]) / ny
            shifted_coords[:, 2] = (shifted_coords[:, 2] + shift[2]) / nz

            all_frac_coords.append(shifted_coords)
            all_labels.append(labels)

        # Combine all coordinates and labels
        combined_frac_coords = np.vstack(all_frac_coords)
        combined_labels = np.concatenate(all_labels)

        # Convert to cartesian coordinates
        cart_coords = self.fractional_to_cartesian(
            combined_frac_coords, supercell_lattice
        )

        # Center if requested
        if center:
            # Calculate the geometric center of the box
            box_center = np.sum(supercell_lattice, axis=0) / 2
            cart_coords -= box_center

        # Create DataFrame with results
        system = pd.DataFrame(
            {
                "site_label": combined_labels,
                "cartesian_x": cart_coords[:, 0],
                "cartesian_y": cart_coords[:, 1],
                "cartesian_z": cart_coords[:, 2],
            }
        )

        # Get FEASST box parameters
        a_vec, b_vec, c_vec = supercell_lattice

        lx, _, _ = a_vec
        xy, ly, _ = b_vec
        xz, yz, lz = c_vec

        box = [lx, ly, lz, xy, xz, yz]
        box = self._reduce_tilt_factors(tuple(box))
        box = tuple([round(v, 14) for v in box])

        vectors = (a_vec, b_vec, c_vec)

        return system, box, vectors

    def create_system(self, unit_cells=(1, 1, 1)):
        """Create a supercell system (wrapper around create_supercell for backward compatibility)."""
        return self.create_supercell(unit_cells)

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
            # "molecules/unitcell_to_cm3stp/g": self._molecules_uc__cm3_g,
            # "molecules/unitcell_to_mol/kg": self._molecules_uc__mol_kg,
        }

        logger.info("Writing metadata to %s.metadata.json", metadata_file_name)
        with open(f"{metadata_file_name}.metadata.json", "w") as metadata_f_out:
            json.dump(metadata, metadata_f_out, indent=4)

        return metadata

    def set_force_field(
        self,
        parameters: Dict,
        by: str = "site_type",
    ) -> None:
        """Set the force field parameters for the framework based on the provided parameters.

        Arguments
        ---------
        parameters : dict
            A dictionary containing force field parameters. Each parameter should be a dictionary with keys
            'sigma', 'epsilon', and 'charge'.
        by : str
            Specifies how to group the parameters. Can be either 'site_type' or 'site_label'.

        Raises
        ------
            ValueError: If 'by' is not one of the allowed values ('site_type', 'site_label').
        """
        if by not in ["site_type", "site_label"]:
            raise ValueError("'by' must be either 'site_type' or 'site_label'")

        site_label_to_type = dict(
            zip(
                self._dataframe["site_label"].values,
                self._dataframe["site_type"].values,
            )
        )

        new_force_field = {}

        for site_label, site_type in site_label_to_type.items():
            lookup_key = site_type if by == "site_type" else site_label

            site_parameters = parameters.get(lookup_key, {})
            existing_parameters = self._force_field.get(site_label, {})

            merged_params = {
                "site_type": site_type,
                "sigma": site_parameters.get("sigma", existing_parameters.get("sigma")),
                "epsilon": site_parameters.get(
                    "epsilon", existing_parameters.get("epsilon")
                ),
                "charge": site_parameters.get(
                    "charge", existing_parameters.get("charge")
                ),
            }

            new_force_field[site_label] = merged_params

        if self._force_field and self._force_field != new_force_field:
            logger.warning("Updating force field parameters.")

        self._force_field = new_force_field

    def group_sites_by_charge(
        self,
        bond_tolerance: float = 0.15,
        small_charge_threshold: float = 0.1,
        relative_threshold: float = 0.15,
        absolute_threshold: float = 0.05,
        charge_bin_size: float = 0.01,
        distance_bin_size: float = 0.2,
        max_cutoff: float = 6.0,
    ):
        """Group atoms in a framework based on their chemical environment and assigns averaged charges to each group.

        Updates both the site labels in the dataframe and the force field parameters.

        Note: This function modifies site labels and charges in the dataframe, preserving original values
        in 'site_original_label' and 'site_original_charge' columns. It also updates the force field
        with averaged charges for each group.

        Args:
        bond_tolerance : float
            bond tolerance in percentage (e.g. 0.15 = 15%). Used in sum of covalent radii to determine
            if two atoms are bonded.
        small_charge_threshold : float
            charges smaller than this value are considered small and use relative threshold for splitting
        relative_threshold : float
            relative threshold for splitting groups with small charges
        absolute_threshold : float
            absolute threshold for splitting groups with large charges
        charge_bin_size : float
            size of the charge bin for grouping atoms
        distance_bin_size : float
            size of the distance bin for fingerprinting atoms (in Angstroms)
        max_cutoff : float
            maximum distance to consider for the supercell creation, should be larger
            than any potential bond (in Angstroms)

        Returns
        -------
        dict
            A dictionary mapping atom labels to their averaged charges
        """

        def is_bonded(element1, element2, distance):
            """Check if two elements are bonded based on distance and their covalent radii.

            Args:
                element1 (str): element symbol of the first atom
                element2 (str): element symbol of the second atom
                distance (float): distance between two atoms in Angstroms

            Returns
            -------
                bool: True if bonded, False otherwise
            """
            r1 = covalent_radii.get(element1, None)
            r2 = covalent_radii.get(element2, None)
            if r1 is None or r2 is None:
                raise ValueError(
                    f"No covalent radius for element {element1} or {element2}"
                )
            return distance <= ((r1 + r2) * (1 + bond_tolerance))

        def should_split_group(group_charges):
            """Determine if a group should be split based on charge variation.

            Args:
                group_charges (numpy.ndarray): Array of charges for atoms in the group

            Returns
            -------
                bool: True if the group should be split, False otherwise
            """
            if len(group_charges) <= 1:
                return False

            mean_charge = np.mean(group_charges)
            std_dev = np.std(group_charges)

            # for small charges, use relative threshold
            if abs(mean_charge) < small_charge_threshold:
                relative_std_dev = std_dev / max(
                    abs(mean_charge), 0.01
                )  # avoid division by zero
                return relative_std_dev > relative_threshold

            # for large charges, use absolute threshold
            else:
                return std_dev > absolute_threshold

        from scipy.spatial import cKDTree

        fractional_coordinates = self._dataframe[
            ["fractional_x", "fractional_y", "fractional_z"]
        ].to_numpy()  # (n_sites, 3)

        # Create a 3x3x3 supercell to handle periodic boundary conditions
        supercell_fractional_coords = []
        supercell_indices = []
        original_atom_indices = np.arange(len(fractional_coordinates))

        # Generate translations in fractional space (just like in create_supercell)
        for i, j, k in product([-1, 0, 1], repeat=3):
            # Shift in fractional space
            shift = np.array([i, j, k])
            shifted_coords = fractional_coordinates + shift

            supercell_fractional_coords.append(shifted_coords)
            supercell_indices.append(original_atom_indices)

        supercell_frac_coords = np.vstack(supercell_fractional_coords)
        supercell_indices = np.concatenate(supercell_indices)
        supercell_coordinates = self.fractional_to_cartesian(supercell_frac_coords)
        cartesian_coordinates = self.fractional_to_cartesian(fractional_coordinates)

        # Update dataframe with cartesian coordinates
        df = self._dataframe.copy()
        df[["cartesian_x", "cartesian_y", "cartesian_z"]] = cartesian_coordinates

        supercell_tree = cKDTree(supercell_coordinates)
        original_tree = cKDTree(cartesian_coordinates)

        # find all potential neighbors for each original atom within the max_cutoff distance
        neighbors = original_tree.query_ball_tree(supercell_tree, r=max_cutoff)

        all_neighbors = {}
        for i in range(len(df)):
            central_element = df.loc[i, "site_type"]
            bonded_atoms = []

            for neighbor_supercell_idx in neighbors[i]:
                neighbor_original_idx = supercell_indices[neighbor_supercell_idx]

                # ignore self-interactions
                if i == neighbor_original_idx and np.allclose(
                    cartesian_coordinates[i],
                    supercell_coordinates[neighbor_supercell_idx],
                ):
                    continue

                distance = np.linalg.norm(
                    cartesian_coordinates[i]
                    - supercell_coordinates[neighbor_supercell_idx]
                )
                neighbor_element = df.loc[neighbor_original_idx, "site_type"]

                if is_bonded(neighbor_element, central_element, distance):
                    bonded_atoms.append((neighbor_original_idx, distance))

            all_neighbors[i] = bonded_atoms

        # generate fingerprints for each atom based on its neighbors
        atom_fingerprints = {}
        for i in range(len(df)):
            central_element = df.loc[i, "site_type"]

            # first neighbors fingerprint
            first_shell = []
            first_shell_data = all_neighbors.get(i, [])
            first_shell_indices = [idx for idx, _ in first_shell_data]

            for neighbor_idx, distance in first_shell_data:
                neighbor_element = df.loc[neighbor_idx, "site_type"]
                binned_distance = (
                    round(distance / distance_bin_size) * distance_bin_size
                )
                first_shell.append((neighbor_element, f"{binned_distance:.2f}"))

            first_shell.sort()

            # second neighbors fingerprint
            second_shell_elements = set()
            for neighbor_idx, _ in first_shell_data:
                for second_neighbor_idx, __ in all_neighbors.get(i, []):
                    if (
                        second_neighbor_idx != i
                        and second_neighbor_idx not in first_shell_indices
                    ):
                        second_shell_elements.add(
                            df.loc[second_neighbor_idx, "site_type"]
                        )

            final_fingerprint = (
                central_element,
                tuple(first_shell),
                tuple(sorted(second_shell_elements)),
            )
            atom_fingerprints[i] = final_fingerprint

        # group by fingerprints
        grouped_atoms_initial = {}
        for atom_idx, fp in atom_fingerprints.items():
            grouped_atoms_initial.setdefault(fp, []).append(atom_idx)

        # refine groups by charge
        grouped_atoms_final = {}
        for fp, indices in grouped_atoms_initial.items():
            group_charges = df.loc[indices, "site_charge"].to_numpy()

            if not should_split_group(group_charges):
                grouped_atoms_final[fp] = indices
            else:
                for idx in indices:
                    charge = df.loc[idx, "site_charge"]
                    charge_bin = round(charge / charge_bin_size)
                    refined_fp = fp + (f"charge_bin_{charge_bin}",)
                    grouped_atoms_final.setdefault(refined_fp, []).append(idx)

        logger.info("Grouped atoms into %d groups.", len(grouped_atoms_final))

        self._dataframe["site_original_charge"] = self._dataframe["site_charge"].copy()
        self._dataframe["site_original_label"] = self._dataframe["site_label"].copy()

        element_counters = {}
        groups = {}
        force_field_parameters = {}

        for fp, indices in grouped_atoms_final.items():
            group_charges = df.loc[indices, "site_charge"]
            average_charge = group_charges.mean()
            std_charge = group_charges.std()

            central_element = fp[0]

            if central_element not in element_counters:
                element_counters[central_element] = 0
            else:
                element_counters[central_element] += 1

            group_id = f"{central_element}{element_counters[central_element]}"

            self._dataframe.loc[indices, "site_charge"] = average_charge
            self._dataframe.loc[indices, "site_label"] = group_id

            groups[group_id] = {
                "average_charge": average_charge,
                "std_charge": std_charge,
                "count": len(indices),
                "atom_labels": df.loc[indices, "site_label"].tolist(),
                "fingerprint": str(fp),
            }

            force_field_parameters[group_id] = {"charge": average_charge}

            # Log group information
            logger.debug(
                "Group %s: %d atoms, charge=%.4f±%.4f",
                group_id,
                groups[group_id]["count"],
                average_charge,
                std_charge,
            )

        self.set_force_field(force_field_parameters, by="site_label")

        return groups

    def write_xyz_file(
        self,
        file_name: str,
        system: pd.DataFrame,
        vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Write system in extxyz file format."""
        n_sites = system.shape[0]
        flat_vectors = np.concatenate(vectors)
        vectors_str = " ".join(f"{x:.10f}" for x in flat_vectors)

        with open(f"{file_name}.xyz", "w") as xyz_file:
            print(n_sites, file=xyz_file)
            print(
                f'Lattice="{vectors_str}" Properties=species:S:1:pos:R:3', file=xyz_file
            )
            print(system.to_string(header=False, index=False), file=xyz_file)

    @staticmethod
    def dl_poly_ewald(cutoff: float, box: tuple, tolerance: float = 0.00001):
        """Calculate `alpha` and `kmax` parameters for Ewald summation.

        Recipe from the DL_POLY Algorithm
        https://doi.org/10.1080/002689798167881
        thanks to Daniel W. Siderius
        """
        eps = min(tolerance, 0.5)
        xi = np.sqrt(np.abs(np.log(eps * cutoff)))
        alpha = np.sqrt(np.abs(np.log(eps * cutoff * xi))) / cutoff
        chi = np.sqrt(-np.log(eps * cutoff * ((2.0 * xi * alpha) ** 2)))
        kmax = [int(0.25 + box[i] * alpha * chi / np.pi) for i in range(3)]

        return alpha, kmax

    def write_fstprt(
        self,
        file_name: str | Path,
        unit_cells: tuple[int, int, int] = (1, 1, 1),
        cutoff: float = 12.8,
        return_metadata: bool = False,
        ewald_tolerance: float = 0.00001,
    ) -> None | dict:
        """Write molecule file with framework for FEASST simulation software."""
        if len(unit_cells) != 3 or not all(isinstance(n, int) for n in unit_cells):
            raise ValueError("`unit_cells` must be three positive integers")

        system, box, vectors = self.create_system(unit_cells)
        net_charge = self.check_net_charge(unit_cells)
        logger.info("Net charge is %e", net_charge)
        # alpha, kmax = self.dl_poly_ewald(
        #     cutoff=cutoff, box=box, tolerance=ewald_tolerance
        # )

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

        file = """# FEASST particle file (https://doi.org/10.18434/M3S095)
#
# Units
# length: Angstrom
# energy: kJ/mol
# charge: elementary

Site Properties

"""
        for site_label, site_parameters in self._force_field.items():
            line = (
                f"{site_label:<3} "
                + f"sigma={site_parameters['sigma']:.5f} "
                + f"epsilon={site_parameters['epsilon']:.8f} "
                + f"cutoff={cutoff:.1f} "
                + f"charge={site_parameters['charge']:.15f}\n"
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
