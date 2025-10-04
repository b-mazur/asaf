"""Tests for the Framework class."""

import json
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from gemmi import cif

from asaf.framework import Framework


@pytest.fixture
def orthorhombic_framework():
    """Create a tiny, orthorhombic framework with 3 sites (C, O, H)."""
    lattice = np.array(
        [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]], dtype=float
    )
    sites = ["C_1", "O_1", "H_1"]
    site_types = ["C", "O", "H"]
    coords = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=float,
    )
    charges = [0.10, -0.20, 0.05]

    return Framework(
        lattice=lattice,
        sites=sites,
        coordinates=coords,
        lattice_as_matrix=True,
        site_types=site_types,
        charges=charges,
    )


@pytest.fixture
def triclinic_framework():
    """Create a framework with a triclinic cell for testing tilt factors."""
    lattice_params = [10.0, 12.0, 14.0, 80.0, 85.0, 95.0]
    sites = ["C_1", "O_1"]
    coords = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ]
    )

    return Framework(
        lattice=lattice_params,
        sites=sites,
        coordinates=coords,
        lattice_as_matrix=False,
    )


class TestFrameworkInitialization:
    def test_init_with_lattice_parameters(self):
        """Test initialization with lattice parameters (a, b, c, alpha, beta, gamma)."""
        fw = Framework(
            lattice=[10.0, 12.0, 14.0, 90.0, 90.0, 90.0],
            sites=["C_1"],
            coordinates=np.array([[0.5, 0.5, 0.5]]),
        )
        assert hasattr(fw, "_lattice")
        assert fw._lattice.shape == (3, 3)
        np.testing.assert_allclose(
            fw._lattice, np.diag([10.0, 12.0, 14.0]), rtol=1e-10, atol=1e-10
        )

    def test_init_with_matrix(self):
        """Test initialization with direct lattice matrix."""
        matrix = np.array(
            [
                [10.0, 0.0, 0.0],
                [1.0, 12.0, 0.0],  # with xy tilt component
                [0.0, 1.0, 14.0],  # with yz tilt component
            ]
        )
        fw = Framework(
            lattice=matrix,
            sites=["C_1"],
            coordinates=np.array([[0.5, 0.5, 0.5]]),
            lattice_as_matrix=True,
        )
        np.testing.assert_array_equal(fw._lattice, matrix)

    def test_init_validation_errors(self):
        """Test validation errors during initialization."""
        # Wrong length when passing parameters (not matrix)
        with pytest.raises(ValueError):
            Framework(
                lattice=[10.0, 12.0, 14.0, 90.0],  # only 4 values
                sites=["C_1"],
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                lattice_as_matrix=False,
            )

        # Wrong matrix shape when lattice_as_matrix=True
        with pytest.raises(ValueError):
            Framework(
                lattice=np.eye(2),  # 2x2 instead of 3x3
                sites=["C_1"],
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                lattice_as_matrix=True,
            )

    def test_default_charges(self):
        """Test that charges default to zero."""
        fw = Framework(
            lattice=[10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            sites=["C_1", "O_1"],
            coordinates=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        )
        assert all(charge == 0.0 for charge in fw._dataframe["site_charge"])

    def test_inferred_site_types(self):
        """Test site_type inference from site labels."""
        fw = Framework(
            lattice=[10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            sites=["C_1", "O_2", "H_3"],
            coordinates=np.array(
                [
                    [0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.2],
                    [0.3, 0.3, 0.3],
                ]
            ),
        )
        assert list(fw._dataframe["site_type"]) == ["C", "O", "H"]


class TestLatticeOperations:
    def test_lattice_parameters_to_matrix(self):
        """Test conversion from lattice parameters to matrix."""
        matrix = Framework.lattice_parameters_to_matrix(
            10.0, 12.0, 14.0, 90.0, 90.0, 90.0
        )
        expected = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 12.0, 0.0],
                [0.0, 0.0, 14.0],
            ]
        )
        np.testing.assert_array_almost_equal(matrix, expected, decimal=12)
        # Non-orthorhombic case
        matrix = Framework.lattice_parameters_to_matrix(
            10.0, 12.0, 14.0, 80.0, 85.0, 95.0
        )
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 0.0
        assert matrix[0, 2] == 0.0
        assert matrix[1, 2] == 0.0
        np.testing.assert_allclose(np.linalg.norm(matrix[0]), 10.0, rtol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(matrix[1]), 12.0, rtol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(matrix[2]), 14.0, rtol=1e-10)

    def test_reduce_tilt_factors(self, triclinic_framework):
        """Test tilt factor reduction to LAMMPS canonical range."""
        # Create a box with out-of-range tilt
        box = (10.0, 10.0, 10.0, 7.0, 2.0, 3.0)  # xy > lx/2
        reduced = Framework._reduce_tilt_factors(box)

        assert reduced[3] == pytest.approx(-3.0)  # 7.0 - 10.0
        assert reduced[4] == pytest.approx(2.0)  # unchanged
        assert reduced[5] == pytest.approx(3.0)  # unchanged
        # Edge case: exactly at lx/2
        box = (10.0, 10.0, 10.0, 5.0, 0.0, 0.0)
        reduced = Framework._reduce_tilt_factors(box)
        assert abs(reduced[3]) == pytest.approx(5.0)
        # Error case: negative box length
        with pytest.raises(ValueError):
            Framework._reduce_tilt_factors((-1.0, 10.0, 10.0, 0.0, 0.0, 0.0))

    def test_fractional_to_cartesian(self, orthorhombic_framework):
        """Test conversion from fractional to Cartesian coordinates."""
        fractional = np.array([[0.1, 0.2, 0.3]])
        expected = np.array([[1.0, 2.4, 4.2]])
        cartesian = orthorhombic_framework.fractional_to_cartesian(fractional)
        np.testing.assert_allclose(cartesian, expected, rtol=1e-10)
        # Test with multiple points
        fractional = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 6.0, 7.0],
                [10.0, 12.0, 14.0],
            ]
        )
        cartesian = orthorhombic_framework.fractional_to_cartesian(fractional)
        np.testing.assert_allclose(cartesian, expected, rtol=1e-10)


class TestFrameworkProperties:
    def test_calculate_framework_mass(self, orthorhombic_framework):
        """Test framework mass calculation."""
        mass = orthorhombic_framework.calculate_framework_mass()
        assert mass > 0.0
        # Should be cached (second call returns same, no exception)
        mass2 = orthorhombic_framework.calculate_framework_mass()
        assert mass2 == pytest.approx(mass)
        # Verify mass is approximately correct
        # C (12) + O (16) + H (1) = 29 g/mol
        assert mass == pytest.approx(29.0, rel=0.1)
        # Verify KeyError on unknown element
        lattice = np.eye(3) * 10.0
        sites = ["Xx_1"]
        coords = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(KeyError):
            Framework(
                lattice, sites, coords, lattice_as_matrix=True
            ).calculate_framework_mass()

    def test_calculate_volume(self, orthorhombic_framework, triclinic_framework):
        """Test unit cell volume calculation."""
        vol = orthorhombic_framework.calculate_framework_unitcell_volume()
        assert vol == pytest.approx(10.0 * 12.0 * 14.0)
        # Triclinic case
        vol = triclinic_framework.calculate_framework_unitcell_volume()
        assert vol > 0.0
        assert vol == pytest.approx(abs(np.linalg.det(triclinic_framework._lattice)))

    def test_conversion_factors(self, orthorhombic_framework):
        """Test unit conversion factor calculations."""
        conv = orthorhombic_framework.calculate_conversion_factors()
        assert conv["molecules_uc__mol_kg"] == pytest.approx(
            34.46136880556896, rel=1e-5, abs=1e-8
        )
        assert conv["molecules_uc__cm3_g"] == pytest.approx(
            772.4160708875229, rel=1e-5, abs=1e-8
        )
        assert conv["molecules_uc__cm3_cm3"] == pytest.approx(
            22.15432861901235, rel=1e-5, abs=1e-8
        )
        # Now with an adsorbate molar mass (e.g., water ~18 g/mol)
        conv2 = orthorhombic_framework.calculate_conversion_factors(
            adsorbate_molar_mass=18.01528
        )
        assert conv2["molecules_uc__g_g"] == pytest.approx(
            0.6208312082155905, rel=1e-5, abs=1e-8
        )
        # Bad input
        with pytest.raises(ValueError):
            orthorhombic_framework.calculate_conversion_factors(
                adsorbate_molar_mass=-1.0
            )

    def test_site_helpers(self, orthorhombic_framework):
        """Test site label and type accessors."""
        labels = orthorhombic_framework.site_labels(as_list=True)
        types = orthorhombic_framework.site_types(as_list=True)
        assert isinstance(labels, list)
        assert isinstance(types, list)
        assert len(labels) == len(types) == 3
        assert set(types) == {"C", "O", "H"}
        # Test Series output
        labels_series = orthorhombic_framework.site_labels(as_list=False)
        types_series = orthorhombic_framework.site_types(as_list=False)
        assert isinstance(labels_series, pd.Series)
        assert isinstance(types_series, pd.Series)


class TestChargeOperations:
    def test_check_net_charge(self, orthorhombic_framework):
        """Test net charge calculation."""
        net = orthorhombic_framework.check_net_charge((1, 1, 1))
        assert isinstance(net, float)
        assert net == pytest.approx(-0.05)
        # With 2x2x2 supercell, charge should scale by 8
        net = orthorhombic_framework.check_net_charge((2, 2, 2))
        assert net == pytest.approx(-0.4)

    def test_reduce_net_charge(self, orthorhombic_framework):
        """Test net charge reduction."""
        orthorhombic_framework.reduce_net_charge()
        net = orthorhombic_framework.check_net_charge((1, 1, 1))
        assert abs(net) < 1e-12  # numerically ~0
        # Check distribution is proportional to charge magnitudes
        charges = orthorhombic_framework._dataframe["site_charge"].to_numpy()
        original = np.array([0.10, -0.20, 0.05])
        total_correction = float(original.sum())  # Original total we need to distribute
        # Corrections should be proportional to absolute charge values
        abs_sum = float(np.abs(original).sum())
        expected = original - (total_correction * (original / abs_sum))
        expected = expected - expected.sum() / expected.size
        # Verify charges are updated correctly (within floating point tolerance)
        np.testing.assert_allclose(charges, expected, rtol=1e-10, atol=1e-10)
        # Test edge case: all charges zero
        fw = Framework(
            lattice=np.eye(3),
            sites=["A_1", "B_1"],
            coordinates=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
            charges=[0, 0],
            lattice_as_matrix=True,
        )
        fw.reduce_net_charge()


class TestSupercellCreation:
    def test_create_supercell(self, orthorhombic_framework):
        """Test supercell creation with various parameters."""
        df, box, vectors = orthorhombic_framework.create_supercell((1, 1, 1))
        assert len(df) == 3
        assert len(box) == 6
        # 2x1x1 should return 6 sites
        df, box, vectors = orthorhombic_framework.create_supercell((2, 1, 1))
        assert len(df) == 6
        assert box[0] == pytest.approx(20.0)
        # 1x1x2 should return 6 sites with z-dimension doubled
        df, box, vectors = orthorhombic_framework.create_supercell((1, 1, 2))
        assert len(df) == 6
        assert box[2] == pytest.approx(28.0)
        # 2x2x2 should return 24 sites (3*2*2*2)
        df, box, vectors = orthorhombic_framework.create_supercell((2, 2, 2))
        assert len(df) == 24
        # Test centered option
        df_center, _, _ = orthorhombic_framework.create_supercell(
            (1, 1, 1), center=True
        )
        df_nocenter, _, _ = orthorhombic_framework.create_supercell(
            (1, 1, 1), center=False
        )
        # Centered should have coordinates shifted by half box size
        center_offset = np.array([5.0, 6.0, 7.0])  # half of 10,12,14
        for i in range(3):
            assert df_center[
                f"cartesian_{chr(ord('x') + i)}"
            ].to_numpy().mean() == pytest.approx(
                df_nocenter[f"cartesian_{chr(ord('x') + i)}"].to_numpy().mean()
                - center_offset[i]
            )

    def test_create_system_compatibility(self, orthorhombic_framework):
        """Test that create_system is a compatible wrapper for create_supercell."""
        df1, box1, vectors1 = orthorhombic_framework.create_supercell((2, 2, 2))
        df2, box2, vectors2 = orthorhombic_framework.create_system((2, 2, 2))

        pd.testing.assert_frame_equal(df1, df2)
        assert box1 == box2
        for v1, v2 in zip(vectors1, vectors2):
            np.testing.assert_array_equal(v1, v2)


class TestForceField:
    def test_set_force_field(self, orthorhombic_framework):
        """Test force field parameter setting."""
        params_by_type = {
            "C": {"sigma": 3.4, "epsilon": 0.1, "charge": 0.0},
            "O": {"sigma": 3.0, "epsilon": 0.2, "charge": -0.2},
        }
        orthorhombic_framework.set_force_field(params_by_type, by="site_type")
        ff = orthorhombic_framework._force_field
        assert ff["C_1"]["sigma"] == pytest.approx(3.4)
        assert ff["O_1"]["epsilon"] == pytest.approx(0.2)
        # Set by site_label (should override)
        params_by_label = {
            "C_1": {"sigma": 3.5, "epsilon": 0.15},
            "O_1": {"charge": -0.25},
        }
        orthorhombic_framework.set_force_field(params_by_label, by="site_label")
        # Check merged parameters
        assert ff["C_1"]["sigma"] == pytest.approx(3.5)  # updated
        assert ff["C_1"]["epsilon"] == pytest.approx(0.15)  # updated
        assert ff["C_1"]["charge"] == pytest.approx(0.0)  # unchanged
        assert ff["O_1"]["sigma"] == pytest.approx(3.0)  # unchanged
        assert ff["O_1"]["epsilon"] == pytest.approx(0.2)  # unchanged
        assert ff["O_1"]["charge"] == pytest.approx(-0.25)  # updated
        # Invalid 'by' value
        with pytest.raises(ValueError):
            orthorhombic_framework.set_force_field({}, by="invalid")


class TestFileIO:
    def test_write_metadata_and_xyz(self, orthorhombic_framework, tmp_path):
        """Test metadata and XYZ file writing."""
        df, box, vectors = orthorhombic_framework.create_supercell((1, 1, 1))
        meta_name = tmp_path / "framework_test"
        # Write metadata and check file
        md = orthorhombic_framework.write_metadata(
            meta_name, box, (1, 1, 1), 12.8, vectors
        )
        assert isinstance(md, dict)
        out_file = Path(str(meta_name) + ".metadata.json")
        assert out_file.exists()
        # Check JSON content
        data = json.loads(out_file.read_text())
        for key in ["box_size", "tilt_factors", "lattice", "unit_cells", "cutoff"]:
            assert key in data
        # Write XYZ file
        xyz_name = tmp_path / "framework_test"
        orthorhombic_framework.write_xyz_file(xyz_name, df, vectors)
        xyz_file = Path(str(xyz_name) + ".xyz")
        assert xyz_file.exists()
        # Verify XYZ content
        text = xyz_file.read_text()
        assert "Lattice=" in text
        lines = text.splitlines()
        assert int(lines[0]) == len(df)
        assert "Properties=species" in lines[1]

    def test_dl_poly_ewald(self, orthorhombic_framework):
        """Test Ewald parameter calculation."""
        _, box, _ = orthorhombic_framework.create_supercell((1, 1, 1))
        alpha, kmax = orthorhombic_framework.dl_poly_ewald(
            cutoff=12.8, box=box, tolerance=1e-5
        )

        assert alpha > 0.0
        assert isinstance(kmax, list)
        assert len(kmax) == 3
        assert all(k >= 1 for k in kmax)
        # Test with different tolerance
        alpha2, kmax2 = orthorhombic_framework.dl_poly_ewald(
            cutoff=12.8, box=box, tolerance=1e-8
        )
        assert any(k2 >= k1 for k1, k2 in zip(kmax, kmax2))

    def test_write_fstprt(self, orthorhombic_framework, tmp_path, monkeypatch):
        """Test FSTPRT file writing."""
        orthorhombic_framework.set_force_field(
            {
                "C": {"sigma": 3.4, "epsilon": 0.1, "charge": 0.0},
                "O": {"sigma": 3.0, "epsilon": 0.2, "charge": -0.2},
                "H": {"sigma": 2.5, "epsilon": 0.05, "charge": 0.2},
            }
        )
        # Write FSTPRT file
        out_path = tmp_path / "framework"
        meta = orthorhombic_framework.write_fstprt(
            out_path, unit_cells=(1, 1, 1), cutoff=12.8, return_metadata=True
        )
        # Check metadata was returned
        assert isinstance(meta, dict)
        # Check files were created
        fstprt_file = Path(str(out_path) + ".fstprt")
        meta_file = Path(str(out_path) + ".metadata.json")
        xyz_file = Path(str(out_path) + ".xyz")
        assert fstprt_file.exists()
        assert meta_file.exists()
        assert xyz_file.exists()
        # Check FSTPRT content
        content = fstprt_file.read_text()
        assert "# FEASST particle file" in content
        assert "Site Properties" in content
        assert "Sites" in content
        # Test invalid unit cells
        with pytest.raises(ValueError):
            orthorhombic_framework.write_fstprt(tmp_path / "bad", unit_cells=(1, 1))


class TestCifImport:
    @pytest.fixture
    def mock_cif_data(self):
        """Create a mock CIF data structure."""
        mock_block = mock.MagicMock()
        mock_block.find_value.side_effect = lambda tag: {
            "_symmetry_space_group_name_H-M": "P 1",
            "_cell_length_a": "10.0",
            "_cell_length_b": "12.0",
            "_cell_length_c": "14.0",
            "_cell_angle_alpha": "90.0",
            "_cell_angle_beta": "90.0",
            "_cell_angle_gamma": "90.0",
        }[tag]
        mock_block.find_loop.side_effect = lambda tag: {
            "_atom_site_label": ["C1", "O1", "H1"],
            "_atom_site_type_symbol": ["C", "O", "H"],
            "_atom_site_fract_x": ["0.1", "0.4", "0.7"],
            "_atom_site_fract_y": ["0.2", "0.5", "0.8"],
            "_atom_site_fract_z": ["0.3", "0.6", "0.9"],
            "_atom_site_charge": ["0.1", "-0.2", "0.1"],
        }[tag]

        mock_cif = mock.MagicMock()
        mock_cif.sole_block.return_value = mock_block
        return mock_cif

    def test_from_cif(self, mock_cif_data, monkeypatch):
        """Test importing framework from a CIF file."""
        monkeypatch.setattr(cif, "read", lambda filename: mock_cif_data)
        # Call from_cif
        fw = Framework.from_cif(Path("dummy.cif"))
        # Verify framework properties
        assert len(fw._dataframe) == 3
        assert list(fw._dataframe["site_label"]) == ["C1", "O1", "H1"]
        assert list(fw._dataframe["site_type"]) == ["C", "O", "H"]
        assert fw._dataframe["site_charge"].sum() == pytest.approx(0.0)
        # Test with remove_site_labels=True
        fw2 = Framework.from_cif(Path("dummy.cif"), remove_site_labels=True)
        assert list(fw2._dataframe["site_label"]) == ["C", "O", "H"]

    def test_from_cif_errors(self, monkeypatch):
        """Test error handling in from_cif."""
        monkeypatch.setattr(
            cif, "read", lambda filename: exec('raise ValueError("Test error")')
        )
        with pytest.raises(ValueError, match="Unable to read CIF"):
            Framework.from_cif(Path("nonexistent.cif"))
        # Mock for wrong symmetry
        mock_wrong_sym = mock.MagicMock()
        mock_block = mock.MagicMock()
        mock_block.find_value.return_value = "P 21/c"
        mock_wrong_sym.sole_block.return_value = mock_block
        monkeypatch.setattr(cif, "read", lambda filename: mock_wrong_sym)
        with pytest.raises(ValueError, match="only P1 symmetry"):
            Framework.from_cif(Path("wrong_sym.cif"))


class TestAdvancedFeatures:
    def test_group_sites_by_charge_basic(self, orthorhombic_framework):
        """Test basic functionality of group_sites_by_charge."""
        orthorhombic_framework._dataframe["site_charge"] = [0.2, -0.3, 0.1]
        # Group sites
        groups = orthorhombic_framework.group_sites_by_charge(
            bond_tolerance=0.15, charge_bin_size=0.05
        )
        # Should have at least one group
        assert isinstance(groups, dict)
        assert len(groups) >= 1
        # Check that dataframe was updated
        assert "site_original_label" in orthorhombic_framework._dataframe.columns
        assert "site_original_charge" in orthorhombic_framework._dataframe.columns
        # Check force field was updated
        assert len(orthorhombic_framework._force_field) >= 1
        # Values should be finite
        for group_id, group_data in groups.items():
            assert math.isfinite(group_data["average_charge"])
            assert group_data["count"] >= 1
