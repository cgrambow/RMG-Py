#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from rmgpy.tools.loader import loadRMGJob

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='INPUT', type=str, nargs=1,
        help='RMG input file')
    parser.add_argument('chemkin', metavar='CHEMKIN', type=str, nargs=1,
        help='Chemkin file')
    parser.add_argument('dictionary', metavar='DICTIONARY', type=str, nargs=1,
        help='RMG dictionary file')
    parser.add_argument('-f', '--foreign', dest='checkDuplicates', action='store_true',
        help='Not an RMG generated Chemkin file (will be checked for duplicates)')
    args = parser.parse_args()
    
    inputFile = os.path.abspath(args.input[0])
    chemkinFile = os.path.abspath(args.chemkin[0])
    dictFile = os.path.abspath(args.dictionary[0])
    checkDuplicates = args.checkDuplicates

    return inputFile, chemkinFile, dictFile, checkDuplicates

def makeSeedMech(inputFile, chemkinFile, dictFile, checkDuplicates=False):
    rmg = loadRMGJob(inputFile, chemkinFile, dictFile, generateImages=False, checkDuplicates=checkDuplicates)
    
    rmg.saveSeedToDatabase = False
    rmg.makeSeedMech(firstTime=True)

def main():
    inputFile, chemkinFile, dictFile, checkDuplicates = parse_arguments()
    makeSeedMech(inputFile, chemkinFile, dictFile, checkDuplicates=checkDuplicates)
    
if __name__=='__main__':
    main()
