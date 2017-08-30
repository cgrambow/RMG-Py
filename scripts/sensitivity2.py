#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs stand-alone sensitivity analysis on an RMG job with extra
functionality.
"""

from __future__ import division

import os.path
import argparse
import csv
import glob
import re

from rmgpy.tools.sensitivity import runSensitivity
from rmgpy.tools.plot import parseCSVData
from rmgpy.chemkin import getSpeciesIdentifier
from rmgpy.solver.liquid import LiquidReactor


################################################################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='INPUT', type=str, nargs=1,
                        help='RMG input file')
    parser.add_argument('chemkin', metavar='CHEMKIN', type=str, nargs=1,
                        help='Chemkin file')
    parser.add_argument('dictionary', metavar='DICTIONARY', type=str, nargs=1,
                        help='RMG dictionary file')
    parser.add_argument('--no-dlim', dest='dlim', action='store_false',
                        help='Turn off diffusion-limited rates for LiquidReactor')
    args = parser.parse_args()

    inputFile = os.path.abspath(args.input[0])
    chemkinFile = os.path.abspath(args.chemkin[0])
    dictFile = os.path.abspath(args.dictionary[0])
    dflag = args.dlim

    return inputFile, chemkinFile, dictFile, dflag


def main():
    inputFile, chemkinFile, dictFile, dflag = parse_arguments()
    rmg = runSensitivity(inputFile, chemkinFile, dictFile, dflag)

    # Loop through all RMG simulation output files
    for simcsv in glob.glob(os.path.join(rmg.outputDirectory, 'solver', 'simulation*.csv')):
        rxnSysIndex = int(re.search(r'\d+', os.path.basename(simcsv)).group()) - 1
        time, dataList = parseCSVData(simcsv)
        speciesList = rmg.reactionModel.core.species
        newDataList = []

        # Convert species names in dataList to species objects because species
        # names in csv file are uninformative without corresponding mechanism
        # file.
        for data in dataList:
            for i, s in enumerate(speciesList):
                if data.label == getSpeciesIdentifier(s):
                    data.species = s
                    break
            if data.species is not None and data.species != 'Volume':
                newDataList.append(data)

        writeMolFracs(rmg, rxnSysIndex, time, newDataList)
        writeElementalComposition(rmg, rxnSysIndex, time, newDataList)

def writeMolFracs(rmg, rxnSysIndex, time, dataList):
    """
    Output a csv file where the first three rows are SMILES, molecular weights,
    and number of radical electrons of the species in an RMG model. After that,
    the columns contain time and species mole fractions.
    """
    smiles = ['SMILES']
    exactMass = ['exact_mass']
    numRadElectrons = ['num_rad_electrons']

    for data in dataList:
        mol = data.species.molecule[0]
        smiles.append(mol.toSMILES())
        exactMass.append(mol.getMolecularWeight() * 1000.0)
        numRadElectrons.append(mol.getRadicalCount())

    path = os.path.join(rmg.outputDirectory, 'solver', 'out_data_{}_species_X.csv'.format(rxnSysIndex + 1))
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE)

        writer.writerow(smiles)
        writer.writerow(exactMass)
        writer.writerow(numRadElectrons)

        for i, t in enumerate(time.data):
            row = [t]
            row.extend(data.data[i] for data in dataList)
            writer.writerow(row)

def writeThermo():
    """
    Not yet implemented because it may not be necessary.
    """
    raise NotImplementedError

def writeElementalComposition(rmg, rxnSysIndex, time, dataList):
    """
    Output two csv files, one with elemental mole fractions of C, H, and O, and
    one with corresponding mass fractions.
    """
    # Remove constant species, because they would falsify results.
    reactionSystem = rmg.reactionSystems[rxnSysIndex]
    if isinstance(reactionSystem, LiquidReactor):
        newDataList = [d for i, d in enumerate(dataList) if i not in reactionSystem.constSPCIndices]
    else:
        newDataList = dataList

    # Extract molecules and set atomic weights
    mols = [d.species.molecule[0] for d in newDataList]
    atomicWeights = {'C': 12.0107, 'H': 1.00794, 'O': 15.9994}  # g/mol

    pathMol = os.path.join(rmg.outputDirectory, 'solver', 'out_data_{}_elemental_mol.csv'.format(rxnSysIndex + 1))
    pathMass = os.path.join(rmg.outputDirectory, 'solver', 'out_data_{}_elemental_mass.csv'.format(rxnSysIndex + 1))
    with open(pathMol, 'w') as fMol, open(pathMass, 'w') as fMass:
        header = ['t', 'C', 'H', 'O']
        writerMol = csv.writer(fMol, delimiter='|', quoting=csv.QUOTE_NONE)
        writerMass = csv.writer(fMass, delimiter='|', quoting=csv.QUOTE_NONE)

        writerMol.writerow(header)
        writerMass.writerow(header)

        # Calculate elemental mole and mass fraction at each time step
        for i, t in enumerate(time.data):
            molFracs = [d.data[i] for d in newDataList]
            elMolFrac, elMassFrac = calculateElementalComposition(molFracs, mols, atomicWeights)

            rowMol = [t, elMolFrac['C'], elMolFrac['H'], elMolFrac['O']]
            rowMass = [t, elMassFrac['C'], elMassFrac['H'], elMassFrac['O']]

            writerMol.writerow(rowMol)
            writerMass.writerow(rowMass)

def calculateElementalComposition(molFracs, mols, atomicWeights):
    """
    Calculate elemental mole and mass fractions for all atoms in the
    atomicWeights dictionary from a list of molecular mole fractions and the
    corresponding molecule objects.
    """
    num = {a: sum(m.getNumAtoms(element=a) * mf for m, mf in zip(mols, molFracs))
           for a in atomicWeights}
    denomMol = sum(m.getNumAtoms(element=a) * mf for m, mf in zip(mols, molFracs)
                   for a in atomicWeights)
    denomMass = sum(m.getMolecularWeight() * 1000.0 * mf for m, mf in zip(mols, molFracs))

    elMolFrac = {a: num[a] / denomMol for a in atomicWeights}
    elMassFrac = {a: w * num[a] / denomMass for a, w in atomicWeights.iteritems()}

    return elMolFrac, elMassFrac

################################################################################

if __name__ == '__main__':
    main()
