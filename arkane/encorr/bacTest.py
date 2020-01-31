#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2019 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
This script contains unit tests for the :mod:`arkane.encorr.bac` module.
"""

import importlib
import os
import tempfile
import unittest

import numpy as np
import pybel

from rmgpy.molecule import Molecule

from arkane.encorr.bac import BAC, _pybel_to_rmg
from arkane.exceptions import BondAdditivityCorrectionError
from arkane.reference import ReferenceDatabase


class TestBAC(unittest.TestCase):
    """
    A class for testing that the BAC class functions properly.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_chem_get = 'ccsd(t)-f12/cc-pvtz-f12'
        cls.model_chem_fit = 'wb97m-v/def2-tzvpd'
        cls.model_chem_nonexisting = 'notamethod/notabasis'

        cls.bac = BAC(cls.model_chem_get)

        cls.tmp_datafile_fd, cls.tmp_datafile_path = tempfile.mkstemp(suffix='.py')

        cls.tmp_melius_params = {
            'atom_corr': {'H': 1.0, 'C': 2.0, 'N': 3.0, 'O': 4.0, 'S': 5.0},
            'bond_corr_length': {'H': 1.0, 'C': 2.0, 'N': 3.0, 'O': 4.0, 'S': 5.0},
            'bond_corr_neighbor': {'H': 1.0, 'C': 2.0, 'N': 3.0, 'O': 4.0, 'S': 5.0},
            'mol_corr': 1.0
        }
        cls.tmp_petersson_params = {'C-H': 1.0, 'C-C': 2.0, 'C=C': 3.0, 'C-O': 4.0}

        # Set molecule, bonds, nums, and coords for testing Petersson and Melius BACs
        smi = 'C=C(OSC=S)C#CC1C(=O)N=CNC1SSC(O)C#N'

        mol = Molecule(smiles=smi)
        cls.bonds = {}
        for bond in mol.get_all_edges():
            symbol = f'{bond.atom1.element.symbol}{BAC.bond_symbols[bond.order]}{bond.atom2.element.symbol}'
            cls.bonds[symbol] = cls.bonds.get(symbol,0)+1

        pybel_mol = pybel.readstring('smi', smi)
        pybel_mol.addh()
        pybel_mol.make3D()
        cls.mol_3d = _pybel_to_rmg(pybel_mol)
        cls.nums = [atom.number for atom in cls.mol_3d.atoms]
        cls.coords = np.array([atom.coords for atom in cls.mol_3d.atoms])

    def test_loading_parameters(self):
        """
        Test that BAC parameters for model chemistries are loaded
        correctly and that errors are raised otherwise.
        """
        self.bac.model_chemistry = self.model_chem_get
        self.bac.bac_type = 'p'
        self.assertIsInstance(self.bac.bacs, dict)

        self.bac.bac_type = 'm'
        self.assertIsNone(self.bac.bacs)

        with self.assertRaises(BondAdditivityCorrectionError):
            self.bac.bac_type = ''

    def test_get_correction(self):
        """
        Test that BAC corrections can be obtained.
        """
        self.bac.model_chemistry = self.model_chem_get
        self.bac.bac_type = 'p'
        corr = self.bac.get_correction(bonds=self.bonds)
        self.assertIsInstance(corr, float)

        # Can use actual Melius parameters once they're available in database
        self.bac.bac_type = 'm'
        corr1 = self.bac._get_melius_correction(coords=self.coords, nums=self.nums, params=self.tmp_melius_params)
        self.assertIsInstance(corr1, float)
        corr2 = self.bac._get_melius_correction(mol=self.mol_3d, params=self.tmp_melius_params)
        self.assertIsInstance(corr2, float)
        self.assertAlmostEqual(corr1, corr2, 5)

        self.bac.model_chemistry = self.model_chem_nonexisting
        with self.assertRaises(BondAdditivityCorrectionError):
            self.bac.get_correction()

    def _clear_bac_data(self):
        self.bac.bacs = None
        self.bac.species = self.bac.ref_data = self.bac.calc_data = self.bac.bac_data = None

    def _check_bac_data(self):
        self.assertIsInstance(self.bac.bacs, dict)
        self.assertIsInstance(self.bac.species, list)
        self.assertIsInstance(self.bac.ref_data, np.ndarray)
        self.assertIsInstance(self.bac.calc_data, np.ndarray)
        self.assertIsInstance(self.bac.bac_data, np.ndarray)
        self.assertTrue(
            len(self.bac.species) == len(self.bac.ref_data) == len(self.bac.calc_data) == len(self.bac.bac_data)
        )
        self.assertLess(self.bac.calculate_stats(self.bac.bac_data).rmse,
                        self.bac.calculate_stats(self.bac.calc_data).rmse)

    def test_fit_petersson(self):
        """
        Test that Petersson BAC parameters can be derived.
        """
        self.bac.model_chemistry = self.model_chem_fit
        self.bac.bac_type = 'p'
        self._clear_bac_data()
        self.bac.fit()

        self._check_bac_data()
        self.assertIn('C-H', self.bac.bacs)

        # Test that database has been loaded
        self.assertIsInstance(self.bac.ref_database, ReferenceDatabase)

        # Test that other instance already has loaded database
        bac = BAC(self.model_chem_fit)
        self.assertIsInstance(bac.ref_database, ReferenceDatabase)

        self.bac.model_chemistry = self.model_chem_nonexisting
        with self.assertRaises(BondAdditivityCorrectionError):
            self.bac.fit()

    def test_fit_melius(self):
        """
        Test that Melius BAC parameters can be derived.
        """
        self.bac.model_chemistry = self.model_chem_fit
        self.bac.bac_type = 'm'
        self._clear_bac_data()

        # With molecular correction, no global opt
        self.bac.fit(fit_mol_corr=True, global_opt=False, lsq_max_nfev=50)
        self._check_bac_data()
        self.assertSetEqual(set(self.bac.bacs.keys()),
                            {'atom_corr', 'bond_corr_length', 'bond_corr_neighbor', 'mol_corr'})
        self.assertNotAlmostEqual(self.bac.bacs['mol_corr'], 0.0)

        # Without molecular correction, with global opt
        self._clear_bac_data()
        self.bac.fit(fit_mol_corr=False, global_opt=True, global_opt_iter=1, lsq_max_nfev=50)
        self._check_bac_data()
        self.assertAlmostEqual(self.bac.bacs['mol_corr'], 0.0)

    def test_calculate_stats(self):
        """
        Test that RMSE and MAE are calculated correctly.
        """
        self.bac.ref_data = None
        with self.assertRaises(BondAdditivityCorrectionError):
            self.bac.calculate_stats(np.array([]))

        self.bac.ref_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            self.bac.calculate_stats(np.array([1.0, 2.0]))

        stats = self.bac.calculate_stats(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
        self.assertLessEqual(stats.mae, stats.rmse)

    def test_write_to_database(self):
        """
        Test that BAC parameters can be written to a file.
        """
        # Check that error is raised when no BACs are available
        self.bac.bacs = None
        with self.assertRaises(BondAdditivityCorrectionError) as e:
            self.bac.write_to_database()
        self.assertIn('No BACs', str(e.exception))

        self.bac.model_chemistry = self.model_chem_get
        self.bac.bac_type = 'p'
        self.bac.bacs = self.tmp_petersson_params

        # Check that error is raised if BACs already exist and overwrite is False
        with self.assertRaises(IOError) as e:
            self.bac.write_to_database(alternate_path=self.tmp_datafile_path)
        self.assertIn('overwrite', str(e.exception))

        # Dynamically set data file as module
        spec = importlib.util.spec_from_file_location(os.path.basename(self.tmp_datafile_path), self.tmp_datafile_path)
        module = importlib.util.module_from_spec(spec)

        # Check that existing Petersson BACs can be overwritten
        self.bac.write_to_database(overwrite=True, alternate_path=self.tmp_datafile_path)
        spec.loader.exec_module(module)  # Load data as module
        self.assertEqual(self.bac.bacs, module.pbac[self.bac.model_chemistry])

        # Check that new Petersson BACs can be written
        self.bac.model_chemistry = self.model_chem_nonexisting
        self.bac.bacs = self.tmp_petersson_params
        self.bac.write_to_database(alternate_path=self.tmp_datafile_path)
        spec.loader.exec_module(module)  # Reload data module
        self.assertEqual(self.bac.bacs, module.pbac[self.bac.model_chemistry])

        # Check that new Melius BACs can be written
        self.bac.bac_type = 'm'
        self.bac.bacs = self.tmp_melius_params
        self.bac.write_to_database(alternate_path=self.tmp_datafile_path)
        spec.loader.exec_module(module)
        self.assertEqual(self.bac.bacs, module.mbac[self.bac.model_chemistry])

    @classmethod
    def tearDownClass(cls):
        os.close(cls.tmp_datafile_fd)
        os.remove(cls.tmp_datafile_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))