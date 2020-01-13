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
This module provides a class for deriving and applying two types of
bond additivity corrections (BACs).

The first type, Petersson-type BACs, are described in:
Petersson et al., J. Chem. Phys. 1998, 109, 10570-10579

The second type, Melius-type BACs, are described in:
Anantharaman and Melius, J. Phys. Chem. A 2005, 109, 1734-1747
"""

import csv
import importlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pybel
import scipy.optimize as optimize

from rmgpy.molecule import Atom, Bond, Molecule, get_element

import arkane.encorr.data as data
from arkane.exceptions import BondAdditivityCorrectionError
from arkane.reference import ReferenceDatabase


class BACJob:
    """
    A representation of an Arkane BAC job. This job is used to fit and
    save bond additivity corrections.
    """

    def __init__(self,
                 model_chemistry: str,
                 bac_type: str = 'p',
                 write_to_database: bool = False,
                 overwrite: bool = False,
                 **kwargs):
        self.model_chemistry = model_chemistry
        self.bac_type = bac_type
        self.write_to_database = write_to_database
        self.overwrite = overwrite
        self.kwargs = kwargs
        self.bac = BAC(model_chemistry, bac_type=bac_type)

    def execute(self, output_directory: str = None, plot: bool = False, jobnum: int = 1):
        """
        Execute the BAC job.

        Args:
            output_directory: Save the results in this directory.
            plot: Save plots of results.
            jobnum: Job number.
        """
        logging.info(f'Running BAC job {jobnum}')
        self.bac.fit(**self.kwargs)

        if output_directory is not None:
            os.makedirs(output_directory, exist_ok=True)
            self.write_output(output_directory, jobnum=jobnum)

            if plot:
                self.plot(output_directory, jobnum=jobnum)

        if self.write_to_database:
            try:
                self.bac.write_to_database(overwrite=self.overwrite)
            except IOError as e:
                logging.warning('Could not write BACs to database. Captured error:')
                logging.warning(str(e))

    def write_output(self, output_directory: str, jobnum: int = 1):
        """
        Save the BACs to the `output.py` file located in
        `output_directory` and save a CSV file of the results.

        Args:
            output_directory: Save the results in this directory.
            jobnum: Job number.
        """
        model_chemistry_formatted = self.model_chemistry.replace('/', '_')
        output_file1 = os.path.join(output_directory, 'output.py')
        output_file2 = os.path.join(output_directory, f'{jobnum}_{model_chemistry_formatted}.csv')
        logging.info(f'Saving results for {self.model_chemistry}...')

        with open(output_file1, 'a') as f:
            stats_before = self.bac.calculate_stats(self.bac.calc_data)
            stats_after = self.bac.calculate_stats(self.bac.bac_data)
            f.write(f'# BAC job {jobnum}: {"Melius" if self.bac.bac_type == "m" else "Petersson"}-type BACs:\n')
            f.write(f'# RMSE/MAE before fitting: {stats_before.rmse:.2f}/{stats_before.mae:.2f} kcal/mol\n')
            f.write(f'# RMSE/MAE after fitting: {stats_after.rmse:.2f}/{stats_after.mae:.2f} kcal/mol\n')
            f.writelines(self.bac.format_bacs())
            f.write('\n')

        with open(output_file2, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Smiles',
                'InChI',
                'Formula',
                'Multiplicity',
                'Charge',
                'Reference Enthalpy',
                'Calculated Enthalpy',
                'Corrected Enthalpy',
                'Source'
            ])
            for spc, ref, calc, bac in zip(self.bac.species, self.bac.ref_data, self.bac.calc_data, self.bac.bac_data):
                writer.writerow([
                    spc.smiles,
                    spc.inchi,
                    spc.formula,
                    spc.multiplicity,
                    spc.charge,
                    f'{ref:.3f}',
                    f'{calc:.3f}',
                    f'{bac:.3f}',
                    spc.get_preferred_source()
                ])

    def plot(self, output_directory: str, jobnum: int = 1):
        """
        Plot the distribution of errors before and after fitting BACs.

        Args:
            output_directory: Save the plots in this directory.
            jobnum: Job number
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        plt.rcParams.update({'font.size': 16})
        model_chemistry_formatted = self.model_chemistry.replace('/', '_')
        fig_path = os.path.join(output_directory, f'{jobnum}_{model_chemistry_formatted}.pdf')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.gca()

        error_before = self.bac.calc_data - self.bac.ref_data
        error_after = self.bac.bac_data - self.bac.ref_data
        _, _, patches = ax.hist(
            (error_before, error_after),
            bins=50,
            label=('before fitting', 'after fitting'),
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_xlabel('Error (kcal/mol)')
        ax.set_ylabel('Count')

        hatches=('////', '----')
        for patch_set, hatch in zip(patches, hatches):
            plt.setp(patch_set, hatch=hatch)
        ax.tick_params(bottom=False)
        ax.set_axisbelow(True)
        ax.grid()
        ax.legend()

        fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)


class BAC:
    """
    A class for deriving and applying bond additivity corrections.
    """

    ref_database = None
    bond_symbols = {1: '-', 2: '=', 3: '#'}
    atom_spins = {
        'H': 0.5, 'C': 1.0, 'N': 1.5, 'O': 1.0, 'F': 0.5,
        'Si': 1.0, 'P': 1.5, 'S': 1.0, 'Cl': 0.5, 'Br': 0.5, 'I': 0.5
    }
    exp_coeff = 3.0  # Melius-type parameter (Angstrom^-1)

    def __init__(self, model_chemistry: str, bac_type: str = 'p'):
        self._model_chemistry = self._bac_type = None  # Set these first to avoid errors in setters
        self.model_chemistry = model_chemistry
        self.bac_type = bac_type

        # Attributes related to fitting BACs for a given model chemistry
        self.species = None  # Reference species
        self.ref_data = None  # Reference enthalpies of formation
        self.calc_data = None  # Calculated enthalpies of formation
        self.bac_data = None  # Calculated data corrected with BACs

    @property
    def bac_type(self) -> str:
        return self._bac_type

    @bac_type.setter
    def bac_type(self, val: str):
        """Check validity and update BACs every time the BAC type is changed."""
        if val not in {'m', 'p'}:
            raise BondAdditivityCorrectionError(f'Invalid BAC type: {val}')
        self._bac_type = val
        self._update_bacs()

    @property
    def model_chemistry(self) -> str:
        return self._model_chemistry

    @model_chemistry.setter
    def model_chemistry(self, val: str):
        """Update BACs every time the model chemistry is changed."""
        self._model_chemistry = val
        self._update_bacs()

    def _update_bacs(self):
        self.bacs = None
        try:
            if self.bac_type == 'm':
                self.bacs = data.mbac[self.model_chemistry]
            elif self.bac_type == 'p':
                self.bacs = data.pbac[self.model_chemistry]
        except KeyError:
            pass

    @classmethod
    def _load_database(cls):
        """Load the reference database"""
        if cls.ref_database is None:
            logging.info('Loading reference database')
            cls.ref_database = ReferenceDatabase()
            cls.ref_database.load()

    def get_correction(self,
                       bonds: Dict[str, int] = None,
                       coords: np.ndarray = None,
                       nums: Iterable[int] = None,
                       multiplicity: int = 1) -> float:
        """
        Returns the bond additivity correction in J/mol.

        There are two bond additivity corrections currently supported.
        Peterson-type corrections can be specified by setting
        `self.bac_type` to 'p'. This will use the `bonds` variable,
        which is a dictionary associating bond types with the number of
        that bond in the molecule.

        The Melius-type BAC is specified with 'm' and utilizes the atom
        coordinates in `coords` and the structure's multiplicity.

        Args:
            bonds: A dictionary of bond types (e.g., 'C=O') with their associated counts.
            coords: A Numpy array of Cartesian molecular coordinates.
            nums: A sequence of atomic numbers.
            multiplicity: The spin multiplicity of the molecule.

        Returns:
            The bond correction to the electronic energy in J/mol.
        """
        if self.bacs is None:
            bac_type_str = 'Melius' if self.bac_type == 'p' else 'Petersson'
            raise BondAdditivityCorrectionError(
                f'Missing {bac_type_str}-type BAC parameters for model chemistry {self.model_chemistry}'
            )

        if self.bac_type == 'm':
            return self._get_melius_correction(coords, nums, multiplicity=multiplicity)
        elif self.bac_type == 'p':
            return self._get_petersson_correction(bonds)

    def _get_petersson_correction(self, bonds: Dict[str, int]) -> float:
        """
        Given the model_chemistry and a dictionary of bonds, return the
        total BAC (should be ADDED to energy).

        Args:
            bonds: Dictionary of bonds with the following format:
                bonds = {
                    'C-H': C-H_bond_count,
                    'C-C': C-C_bond_count,
                    'C=C': C=C_bond_count,
                    ...
                }

        Returns:
            Petersson-type bond additivity correction in J/mol.
        """

        # Sum up corrections for all bonds
        bac = 0.0
        for symbol, count in bonds.items():
            if symbol in self.bacs:
                bac += count * self.bacs[symbol]
            else:
                symbol_flipped = ''.join(re.findall('[a-zA-Z]+|[^a-zA-Z]+', symbol)[::-1])  # Check reversed symbol
                if symbol_flipped in self.bacs:
                    bac += count * self.bacs[symbol_flipped]
                else:
                    logging.warning(f'Ignored unknown bond type {symbol}.')

        return bac * 4184.0  # Convert kcal/mol to J/mol

    def _get_melius_correction(self,
                               coords: np.ndarray = None,
                               nums: Iterable[int] = None,
                               mol: Molecule = None,
                               multiplicity: int = 1,
                               params: Dict[str, Union[float, Dict[str, float]]] = None) -> float:
        """
        Given the model chemistry, molecular coordinates, atomic numbers,
        and dictionaries of BAC parameters, return the total BAC
        (should be SUBTRACTED from energy).

        Note that a molecular correction term other than 0 destroys the size
        consistency of the quantum chemistry method. This correction also
        requires the multiplicity of the molecule.

        Args:
            coords: Numpy array of Cartesian atomic coordinates.
            nums: Sequence of atomic numbers.
            mol: RMG-Py molecule.
            multiplicity: Multiplicity of the molecule.
            params: Optionally provide parameters other than those stored in self.

        Returns:
            Melius-type bond additivity correction in J/mol.
        """
        if params is None:
            params = self.bacs
        atom_corr = params['atom_corr']
        bond_corr_length = params['bond_corr_length']
        bond_corr_neighbor = params['bond_corr_neighbor']
        mol_corr = params.get('mol_corr', 0.0)

        # Get single-bonded RMG molecule
        if mol is None:
            mol = _geo_to_mol(nums, coords)

        # Molecular correction
        spin = 0.5 * (multiplicity - 1)
        bac_mol = mol_corr * (spin - sum(self.atom_spins[atom.element.symbol] for atom in mol.atoms))

        # Atomic correction
        bac_atom = sum(atom_corr[atom.element.symbol] for atom in mol.atoms)

        # Bond correction
        bac_bond = 0.0
        for bond in mol.get_all_edges():
            atom1 = bond.atom1
            atom2 = bond.atom2
            symbol1 = atom1.element.symbol
            symbol2 = atom2.element.symbol

            # Bond length correction
            length_corr = (bond_corr_length[symbol1] * bond_corr_length[symbol2]) ** 0.5
            length = np.linalg.norm(atom1.coords - atom2.coords)
            bac_bond += length_corr * np.exp(-self.exp_coeff * length)

            # Neighbor correction
            for other_atom, other_bond in mol.get_bonds(atom1).items():  # Atoms adjacent to atom1
                if other_bond is not bond:
                    other_symbol = other_atom.element.symbol
                    bac_bond += bond_corr_neighbor[symbol1] + bond_corr_neighbor[other_symbol]
            for other_atom, other_bond in mol.get_bonds(atom2).items():  # Atoms adjacent to atom2
                if other_bond is not bond:
                    other_symbol = other_atom.element.symbol
                    bac_bond += bond_corr_neighbor[symbol2] + bond_corr_neighbor[other_symbol]

        return (bac_mol + bac_atom + bac_bond) * 4184.0  # Convert kcal/mol to J/mol

    def fit(self, **kwargs):
        """
        Fits bond additivity corrections using calculated and reference
        data available in the RMG database. The resulting BACs stored
        in self.bacs will be based on kcal/mol.

        Args:
            kwargs: Keyword arguments for fitting Melius-type BACs (see self._fit_melius).
        """
        self._load_database()  # Will only be loaded the first time that self.fit is called

        self.species = self.ref_database.extract_model_chemistry(self.model_chemistry, as_error_canceling_species=False)
        if not self.species:
            raise BondAdditivityCorrectionError(f'No species available for {self.model_chemistry} model chemistry')

        # Obtain data in kcal/mol
        self.ref_data = np.array([spc.get_reference_enthalpy().h298.value_si / 4184.0 for spc in self.species])
        self.calc_data = np.array(
            [spc.calculated_data[self.model_chemistry].thermo_data.H298.value_si / 4184.0 for spc in self.species]
        )

        if self.bac_type == 'm':
            logging.info(f'Fitting Melius-type BACs for {self.model_chemistry}...')
            self._fit_melius(**kwargs)
        elif self.bac_type == 'p':
            logging.info(f'Fitting Petersson-type BACs for {self.model_chemistry}...')
            self._fit_petersson()

        stats_before = self.calculate_stats(self.calc_data)
        stats_after = self.calculate_stats(self.bac_data)
        logging.info(f'RMSE/MAE before fitting: {stats_before.rmse:.2f}/{stats_before.mae:.2f} kcal/mol')
        logging.info(f'RMSE/MAE after fitting: {stats_after.rmse:.2f}/{stats_after.mae:.2f} kcal/mol')

    def _fit_petersson(self):
        """
        Fit Petersson-type BACs.
        """
        mols = [Molecule().from_adjacency_list(spc.adjacency_list) for spc in self.species]

        def get_features(_mol: Molecule) -> Dict[str, int]:
            """Given a molecule, extract the number of bonds of each type."""
            _features = {}
            for bond in _mol.get_all_edges():
                symbols = [bond.atom1.element.symbol, bond.atom2.element.symbol]
                symbols.sort()  # Ensure that representation is invariant to order of atoms in bond
                symbol = symbols[0] + self.bond_symbols[bond.order] + symbols[1]
                _features[symbol] = _features.get(symbol, 0) + 1
            return _features

        features = [get_features(mol) for mol in mols]
        feature_keys = list({k for f in features for k in f})
        feature_keys.sort()

        def make_feature_mat(_features: List[Dict[str, int]]) -> np.ndarray:
            _x = np.zeros((len(_features), len(feature_keys)))
            for idx, f in enumerate(_features):
                flist = []
                for k in feature_keys:
                    try:
                        flist.append(f[k])
                    except KeyError:
                        flist.append(0.0)
                _x[idx] = np.array(flist)
            return _x

        def lin_reg(_x: np.ndarray, _y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            _w = np.linalg.solve(np.dot(_x.T, _x), np.dot(_x.T, _y))
            _ypred = np.dot(_x, _w)
            return _w, _ypred

        x = make_feature_mat(features)
        y = self.ref_data - self.calc_data
        w, ypred = lin_reg(x, y)

        self.bac_data = self.calc_data + ypred
        self.bacs = {fk: wi for fk, wi in zip(feature_keys, w)}

    def _fit_melius(self,
                    fit_mol_corr: bool = False,
                    global_opt: bool = True,
                    global_opt_iter: int = 15,
                    minimizer_maxiter: int = 100):
        """
        Fit Melius-type BACs.

        Args:
            fit_mol_corr: Also fit molecular correction term.
            global_opt: Perform a global optimization.
            global_opt_iter: Number of iterations for the global minimization.
            minimizer_maxiter: Maximum number of iterations for the SLSQP minimizer.
        """
        conformers = [spc.calculated_data[self.model_chemistry].conformer for spc in self.species]
        geos = [(conformer.number.value.astype(int), conformer.coordinates.value) for conformer in conformers]
        mols = [_geo_to_mol(*geo) for geo in geos]

        all_atom_symbols = list({atom.element.symbol for mol in mols for atom in mol.atoms})
        all_atom_symbols.sort()
        nelements = len(all_atom_symbols)
        low, high = -1e6, 1e6  # Arbitrarily large, just so that we can use bounds in global minimization

        # Specify initial guess.
        # Order of parameters is atom_corr, bond_corr_length, bond_corr_neighbor (, mol_corr)
        # where atom_corr are the atomic corrections, bond_corr_length are the bondwise corrections
        # due to bond lengths (bounded by 0 below), bond_corr_neighbor are the bondwise corrections
        # due to neighboring atoms, and mol_corr (optional) is a molecular correction.
        if fit_mol_corr:
            w0 = np.zeros(3 * nelements + 1) + 1e-6
            wmin = [low] * nelements + [0] * nelements + [low] * nelements + [low]
            wmax = [high] * (3 * nelements + 1)
        else:
            w0 = np.zeros(3 * nelements) + 1e-6
            wmin = [low] * nelements + [0] * nelements + [low] * nelements
            wmax = [high] * 3 * nelements
        bounds = [(lo, hi) for lo, hi in zip(wmin, wmax)]

        class RandomDisplacementBounds:
            """Random displacement with bounds"""
            def __init__(self, stepsize: float = 0.5):
                self.xmin = wmin
                self.xmax = wmax
                self.stepsize = stepsize

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """Take a random step but ensure the new position is within the bounds"""
                while True:
                    xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
                    if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                        break
                return xnew

        def get_params(_w: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
            _atom_corr = dict(zip(all_atom_symbols, _w[:nelements]))
            _bond_corr_length = dict(zip(all_atom_symbols, _w[nelements:2 * nelements]))
            _bond_corr_neighbor = dict(zip(all_atom_symbols, _w[2 * nelements:3 * nelements]))
            _mol_corr = _w[3 * nelements] if fit_mol_corr else 0.0
            return dict(
                atom_corr=_atom_corr,
                bond_corr_length=_bond_corr_length,
                bond_corr_neighbor=_bond_corr_neighbor,
                mol_corr=_mol_corr
            )

        def get_bac_data(_w: np.ndarray) -> np.ndarray:
            corr = np.array(
                [self._get_melius_correction(mol=mol, multiplicity=spc.multiplicity, params=get_params(_w)) / 4184.0
                 for mol, spc in zip(mols, self.species)]
            )
            return self.calc_data - corr  # Need negative sign here

        def objfun(_w: np.ndarray) -> Union[float, np.ndarray]:
            """Least-squares objective function"""
            bac_data = get_bac_data(_w)
            diff = self.ref_data - bac_data
            return np.dot(diff, diff) / len(self.ref_data)

        # SLSQP minimization is a little faster
        minimizer_kwargs = dict(method='SLSQP', bounds=bounds, options={'disp': True, 'maxiter': minimizer_maxiter})
        if global_opt:
            take_step = RandomDisplacementBounds()
            res = optimize.basinhopping(
                objfun, w0, niter=global_opt_iter, minimizer_kwargs=minimizer_kwargs, take_step=take_step, disp=True
            )
        else:
            res = optimize.minimize(objfun, w0, **minimizer_kwargs)
        w = res.x

        self.bac_data = get_bac_data(w)
        self.bacs = get_params(w)

    def calculate_stats(self, calc_data: np.ndarray) -> 'Stats':
        """
        Calculate RMSE and MAE based on stored reference data.

        Args:
            calc_data: Calculated data in same order as self.ref_data.

        Returns:
            Data class with `rmse` and `mae` attributes.
        """
        if self.ref_data is None:
            raise BondAdditivityCorrectionError('Fit BACs before calculating statistics!')

        diff = self.ref_data - calc_data
        rmse = np.sqrt(np.dot(diff, diff) / len(self.ref_data))
        mae = np.sum(np.abs(diff)) / len(self.ref_data)

        return Stats(rmse, mae)

    def write_to_database(self, overwrite: bool = False, alternate_path: str = None):
        """
        Write BACs to data.py.

        Args:
            overwrite: Overwrite existing BACs.
            alternate_path: Write BACs to this path instead.
        """
        if self.bacs is None:
            raise BondAdditivityCorrectionError('No BACs available for writing')

        data_path = os.path.abspath(data.__file__)
        with open(data_path) as f:
            lines = f.readlines()

        bacs_formatted = self.format_bacs(indent=True)

        bac_dict = data.mbac if self.bac_type == 'm' else data.pbac
        keyword = 'mbac' if self.bac_type == 'm' else 'pbac'
        has_entries = bool(data.mbac) if self.bac_type == 'm' else bool(data.pbac)

        # Add new BACs to file without changing existing formatting
        for i, line in enumerate(lines):
            if keyword in line:
                if has_entries:
                    if self.model_chemistry in bac_dict:
                        if overwrite:
                            # Does not overwrite comments
                            for j, line2 in enumerate(lines[i:]):
                                if self.model_chemistry in line2:
                                    del_idx_start = i + j
                                elif line2.rstrip() == '    },':  # Can't have comment after final brace
                                    del_idx_end = i + j + 1
                            if (lines[del_idx_start-1].lstrip().startswith('#')
                                    or lines[del_idx_end+1].lstrip().startswith('#')):
                                logging.warning('There may be left over comments from previous BACs')
                            lines[del_idx_start:del_idx_end] = bacs_formatted
                        else:
                            raise IOError(
                                f'{self.model_chemistry} model chemistry already exists. Set `overwrite` to True.'
                            )
                    else:
                        lines[(i+1):(i+1)] = ['\n'] + bacs_formatted
                else:
                    lines[i] = f'{keyword} = {{\n'
                    lines[(i+1):(i+1)] = ['\n'] + bacs_formatted + ['\n}\n']
                break

        with open(data_path if alternate_path is None else alternate_path, 'w') as f:
            f.writelines(lines)

        # Reload data to update BAC dictionaries
        if alternate_path is None:
            importlib.reload(data)

    def format_bacs(self, indent=False):
        """
        Obtain a list of nicely formatted BACs suitable for writelines.

        Args:
            indent: Indent each line for printing in data.py.

        Returns:
            Formatted list of BACs.
        """
        bacs_formatted = json.dumps(self.bacs, indent=4).replace('"', "'").split('\n')
        bacs_formatted[0] = f"'{self.model_chemistry}': " + bacs_formatted[0]
        bacs_formatted[-1] += ','
        bacs_formatted = [e + '\n' for e in bacs_formatted]
        if indent:
            bacs_formatted = ['    ' + e for e in bacs_formatted]
        return bacs_formatted


@dataclass
class Stats:
    """Small class to store BAC fitting statistics"""
    rmse: Union[float, np.ndarray]
    mae: Union[float, np.ndarray]


def _geo_to_mol(nums: Iterable[int], coords: np.ndarray) -> Molecule:
    """
    Convert molecular geometry specified by atomic coordinates and
    atomic numbers to RMG molecule.

    Use Open Babel for most cases because it's better at recognizing
    long bonds. Use RMG for hydrogen because Open Babel can't do it for
    mysterious reasons.
    """
    if list(nums) == [1, 1]:
        mol = Molecule()
        mol.from_xyz(nums, coords)
    else:
        symbols = [get_element(int(n)).symbol for n in nums]
        xyz = f'{len(symbols)}\n\n'
        xyz += '\n'.join(f'{s}  {c[0]: .10f}  {c[1]: .10f}  {c[2]: .10f}' for s, c in zip(symbols, coords))
        mol = pybel.readstring('xyz', xyz)
        mol = _pybel_to_rmg(mol)
    return mol


def _pybel_to_rmg(pybel_mol: pybel.Molecule) -> Molecule:
    """
    Convert Pybel molecule to RMG molecule but ignore charge,
    multiplicity, and bond orders.
    """
    mol = Molecule()
    for pybel_atom in pybel_mol:
        element = get_element(pybel_atom.atomicnum)
        atom = Atom(element=element, coords=np.array(pybel_atom.coords))
        mol.vertices.append(atom)
    for obbond in pybel.ob.OBMolBondIter(pybel_mol.OBMol):
        begin_idx = obbond.GetBeginAtomIdx() - 1  # Open Babel indexes atoms starting at 1
        end_idx = obbond.GetEndAtomIdx() - 1
        bond = Bond(mol.vertices[begin_idx], mol.vertices[end_idx])
        mol.add_bond(bond)
    return mol
