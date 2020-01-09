#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
This module provides methods for applying Melius-type bond additivity
corrections (M-BAC) as described in:
Anantharaman and Melius, J. Phys. Chem. A 2005, 109, 1734-1747
"""

import numpy as np
import pybel

from rmgpy.molecule import Molecule, Atom, Bond, getElement

from arkane.exceptions import BondAdditivityCorrectionError
import arkane.encorr.data as data

atom_spins = {
    'H': 0.5, 'C': 1.0, 'N': 1.5, 'O': 1.0, 'F': 0.5, 'Si': 1.0, 'P': 1.5, 'S': 1.0, 'Cl': 0.5, 'Br': 0.5, 'I': 0.5
}


def get_bac(model_chemistry, coords=None, nums=None, mol=None, multiplicity=1, params=None):
    """
    Given the model chemistry, molecular coordinates, atomic numbers,
    and dictionaries of BAC parameters, return the total BAC
    (should be SUBTRACTED from energy).

    Note that a molecular correction term other than 0 destroys the size
    consistency of the quantum chemistry method. This correction also
    requires the multiplicity of the molecule.
    """
    alpha = 3.0  # Angstrom^-1

    # Get BAC parameters
    if params is None:
        try:
            params = data.mbac[model_chemistry]
        except KeyError:
            raise BondAdditivityCorrectionError(
                'Missing Melius-type BAC parameters for model chemistry {}'.format(model_chemistry)
            )
    atom_corr = params['atom_corr']
    bond_corr_length = params['bond_corr_length']
    bond_corr_neighbor = params['bond_corr_neighbor']
    mol_corr = params['mol_corr']

    # Get single-bonded RMG molecule
    if mol is None:
        mol = geo_to_mol(coords, nums)

    # Molecular correction
    spin = 0.5 * (multiplicity - 1)
    bac_mol = mol_corr * (spin - sum(atom_spins[atom.element.symbol] for atom in mol.atoms))

    # Atomic correction
    bac_atom = sum(atom_corr[atom.element.symbol] for atom in mol.atoms)

    # Bond correction
    bac_bond = 0.0
    for bond in mol.getAllEdges():
        atom1 = bond.atom1
        atom2 = bond.atom2
        symbol1 = atom1.element.symbol
        symbol2 = atom2.element.symbol

        # Bond length correction
        length_corr = (bond_corr_length[symbol1] * bond_corr_length[symbol2]) ** 0.5
        length = np.linalg.norm(atom1.coords - atom2.coords)
        bac_bond += length_corr * np.exp(-alpha * length)

        # Neighbor correction
        for other_atom, other_bond in mol.getBonds(atom1).iteritems():  # Atoms adjacent to atom1
            if other_bond is not bond:
                other_symbol = other_atom.element.symbol
                bac_bond += bond_corr_neighbor[symbol1] + bond_corr_neighbor[other_symbol]
        for other_atom, other_bond in mol.getBonds(atom2).iteritems():  # Atoms adjacent to atom2
            if other_bond is not bond:
                other_symbol = other_atom.element.symbol
                bac_bond += bond_corr_neighbor[symbol2] + bond_corr_neighbor[other_symbol]

    return (bac_mol + bac_atom + bac_bond) * 4184.0  # Convert kcal/mol to J/mol


def get_grad(model_chemistry, coords=None, nums=None, mol=None, multiplicity=1, mol_corr=True, params=None):
    """Calculate gradient of BAC"""

    # TODO: Plenty of opportunities for caching values, only have to calculate linear terms once

    alpha = 3.0  # Angstrom^-1
    eps = 1e-8

    # Get BAC parameters
    if params is None:
        try:
            params = data.mbac[model_chemistry]
        except KeyError:
            raise BondAdditivityCorrectionError(
                'Missing Melius-type BAC parameters for model chemistry {}'.format(model_chemistry)
            )
    bond_corr_length = params['bond_corr_length']
    symbols = sorted(bond_corr_length.keys())

    # Get single-bonded RMG molecule
    if mol is None:
        mol = geo_to_mol(coords, nums)

    # Atomic terms
    symbols_in_mol = {atom.element.symbol for atom in mol.atoms}
    grad_atom = [1.0 if symbol in symbols_in_mol else 0.0 for symbol in symbols]

    # Bondwise terms
    grad_length = [0] * len(symbols)
    grad_neighbor = [0] * len(symbols)
    bonds = mol.getAllEdges()

    for i, symbol in enumerate(symbols):
        for bond in bonds:
            atom1 = bond.atom1
            atom2 = bond.atom2
            symbol1 = atom1.element.symbol
            symbol2 = atom2.element.symbol

            # Bond length terms
            prefactor = 0
            if symbol == symbol1 == symbol2:
                prefactor = 1
            elif symbol == symbol1 and symbol != symbol2:
                # prefactor = ((bond_corr_length[symbol] * bond_corr_length[symbol2]) ** 0.5
                #              / (2.0 * bond_corr_length[symbol]))
                prefactor = (bond_corr_length[symbol2] /
                             (2.0 * (bond_corr_length[symbol] * bond_corr_length[symbol2] + eps) ** 0.5))
            elif symbol != symbol1 and symbol == symbol2:
                # prefactor = ((bond_corr_length[symbol] * bond_corr_length[symbol1]) ** 0.5
                #              / (2.0 * bond_corr_length[symbol]))
                prefactor = (bond_corr_length[symbol1] /
                             (2.0 * (bond_corr_length[symbol] * bond_corr_length[symbol1] + eps) ** 0.5))

            if prefactor:
                length = np.linalg.norm(atom1.coords - atom2.coords)
                length_exp = np.exp(-alpha * length)
                grad_length[i] += prefactor * length_exp

            # Bond neighbor terms
            if symbol == symbol1 == symbol2:
                nneighbor1 = sum(1 for other_bond in mol.getBonds(atom1).values() if other_bond is not bond)
                nneighbor2 = sum(1 for other_bond in mol.getBonds(atom2).values() if other_bond is not bond)
                grad_neighbor[i] += nneighbor1 + nneighbor2
            elif symbol == symbol1 and symbol != symbol2:
                nneighbor1 = sum(1 for other_bond in mol.getBonds(atom1).values() if other_bond is not bond)
                grad_neighbor[i] += nneighbor1
            elif symbol != symbol1 and symbol == symbol2:
                nneighbor2 = sum(1 for other_bond in mol.getBonds(atom2).values() if other_bond is not bond)
                grad_neighbor[i] += nneighbor2

    if mol_corr:
        # Molecular term
        spin = 0.5 * (multiplicity - 1)
        grad_mol = [spin - sum(atom_spins[atom.element.symbol] for atom in mol.atoms)]
        return np.array(grad_atom + grad_length + grad_neighbor + grad_mol)  # UNITS BASED ON kcal/mol
    else:
        return np.array(grad_atom + grad_length + grad_neighbor)  # UNITS BASED ON kcal/mol


def geo_to_mol(coords, nums):
    """
    Convert molecular geometry specified by atomic coordinates and
    atomic numbers to RMG molecule.

    Use Open Babel for most cases because it's better at recognizing
    long bonds. Use RMG for hydrogen because Open Babel can't do it for
    mysterious reasons.
    """
    if list(nums) == [1, 1]:
        mol = Molecule()
        mol.fromXYZ(nums, coords)
    else:
        xyz = '{}\n\n'.format(len(nums))
        xyz += '\n'.join('{0}  {1[0]: .10f}  {1[1]: .10f}  {1[2]: .10f}'.format(n, c) for n, c in zip(nums, coords))
        mol = pybel.readstring('xyz', xyz)
        mol = pybel_to_rmg(mol)
    return mol


def pybel_to_rmg(pybel_mol):
    """
    Convert Pybel molecule to RMG molecule but ignore charge,
    multiplicity, and bond orders.
    """
    mol = Molecule()
    for pybel_atom in pybel_mol:
        element = getElement(pybel_atom.atomicnum)
        atom = Atom(element=element, coords=np.array(pybel_atom.coords))
        mol.vertices.append(atom)
    for obbond in pybel.ob.OBMolBondIter(pybel_mol.OBMol):
        begin_idx = obbond.GetBeginAtomIdx() - 1  # Open Babel indexes atoms starting at 1
        end_idx = obbond.GetEndAtomIdx() - 1
        bond = Bond(mol.vertices[begin_idx], mol.vertices[end_idx])
        mol.addBond(bond)
    return mol
