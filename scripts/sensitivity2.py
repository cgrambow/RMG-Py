#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs a stand-alone simulation (including sensitivity analysis if
specified in the input file) on an RMG job with extra functionality.
"""

import os.path
import argparse

from rmgpy.tools.simulate import run_simulation


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


################################################################################

if __name__ == '__main__':
    main()
