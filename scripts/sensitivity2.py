#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs stand-alone sensitivity analysis on an RMG job with extra
functionality.
"""

import os.path
import argparse
import glob
import re

from rmgpy.tools.sensitivity import runSensitivity
from rmgpy.tools.plot import parseCSVData


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
    for simcsv in glob.glob(os.path.join(rmg.outputDirectory, 'simulation*.csv')):
        rxnSysIndex = re.search(r'\d+', os.path.basename(simcsv)).group()
        time, dataList = parseCSVData(simcsv)
        speciesList = rmg.reactionModel.core.species[:]

        # Convert species names in dataList to species objects because species
        # names in csv file are uninformative without corresponding mechanism
        # file.
        for data in dataList:
            for i, s in enumerate(speciesList):
                if data.label == s.label:
                    data.species = s.pop[i]
                    break

        writeMolFracs(time, dataList)
        writeThermo(time, dataList)
        writeElementalComposition(time, dataList)

def writeMolFracs(time, dataList):
    pass

def writeThermo(time, dataList):
    pass

def writeElementalComposition(time, dataList):
    pass

################################################################################

if __name__ == '__main__':
    main()
