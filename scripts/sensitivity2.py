#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs a stand-alone simulation (including sensitivity analysis if
specified in the input file) on an RMG job with extra functionality.
"""

import os.path
import argparse
import csv
import glob
import re

from rmgpy.tools.simulate import run_simulation
from rmgpy.tools.plot import parseCSVData
from rmgpy.chemkin import getSpeciesIdentifier


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
    parser.add_argument('-f', '--foreign', dest='checkDuplicates', action='store_true',
                        help='Not an RMG generated Chemkin file (will be checked for duplicates)')
    args = parser.parse_args()

    inputFile = os.path.abspath(args.input[0])
    chemkinFile = os.path.abspath(args.chemkin[0])
    dictFile = os.path.abspath(args.dictionary[0])
    dflag = args.dlim
    checkDuplicates = args.checkDuplicates

    return inputFile, chemkinFile, dictFile, dflag, checkDuplicates


def main():
    inputFile, chemkinFile, dictFile, dflag, checkDuplicates = parse_arguments()
    rmg = run_simulation(inputFile, chemkinFile, dictFile, diffusionLimited=dflag, checkDuplicates=checkDuplicates)

    # Loop through all RMG simulation output files
    for simcsv in glob.glob(os.path.join(rmg.outputDirectory, 'solver', 'simulation*.csv')):
        rxnSysIndex = re.search(r'\d+', os.path.basename(simcsv)).group()
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
        writeElementalComposition(time, newDataList)

def writeMolFracs(rmg, rxnSysIndex, time, dataList):
    """
    Output a csv file where the first three rows are SMILES, molecular weights,
    and number of radical electrons of the species in an RMG model. After that,
    the columns contain time and species mole fractions.
    """
    smiles = ['SMILES']
    exactMass = ['exact_mass']
    numRadElectrons = ['num_rad_electrons']

    exactMasses = {'H': 1.007825, 'C': 12., 'N': 14.003074, 'O': 15.994915, 'He': 4.002603, 'Ne': 19.992440, 'Ar': 39.962383}  # u

    for data in dataList:
        mol = data.species.molecule[0]
        smiles.append(mol.toSMILES())
        exactMass.append(sum(exactMasses[atom.symbol] for atom in mol.atoms))
        numRadElectrons.append(mol.getRadicalCount())

    path = os.path.join(rmg.outputDirectory, 'solver', 'out_data_{}_species_X.csv'.format(rxnSysIndex))
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE)

        writer.writerow(smiles)
        writer.writerow(exactMass)
        writer.writerow(numRadElectrons)

        for i, t in enumerate(time.data):
            row = [t]
            row.extend(data.data[i] for data in dataList)
            writer.writerow(row)

def writeThermo(time, dataList):
    pass

def writeElementalComposition(time, dataList):
    pass

################################################################################

if __name__ == '__main__':
    main()
