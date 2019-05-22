'''
Project description;;;
;;;;;;;;;;;;;;;;;;;;;;

    The project is called MineSoft (short for Mining Software). 
    The project deals with knowledge extraction from software repositories.
    There are two techniques implemented here to extract knowledge from the code:

	(1) SWordNet: Inferring semantically related words from software context, ESE Journal 2014
	(2) VerbNet: Automatically Mining Software-Based,Semantically-Similar Words from Comment-Code Mappings, MSR 2013

    Mainly you can extract pairs of semantically related words from the corpus (repositories)

How to run this script;;;
;;;;;;;;;;;;;;;;;;;;;;;;;

        This main script is the entry point for project. Usually called as:

        python3.4 main.py [-h] -method method_name -input input_dir -output output_dir

        -h, --help            show this help message and exit
        -method method_name, --method method_name
                            which method you want to use (either swordnet or verbnet)
        -input input_dir, --input input_dir
                            which directory is input (e.g. eclipse repo root path)
        -output output_dir, --output output_dir
                            which directory is output (i.e. where pairs will be stored)


'''

__author__ = ["Shayan Ali Akbar"]
__email__ =  ["sakbar@purdue.edu"]

import sys, os
import argparse
import glob
import fnmatch
import nltk
import regex
import concepts
import string
import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swordnet import *
from verbnet import *

class MyArgParser(argparse.ArgumentParser):

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def main(argv):

    # start up message
    startup_message = ("\n\n",
                       "****************************************************************\n",
                       "Welcome to the MineSoft (Mining Software) tool \n",
                       "This tool extracts knowledge from software repository\n",
                       "People working on this tool: Shayan Ali Akbar (sakbar at purdue.edu)\n",
                       "****************************************************************\n\n")
    print (''.join(startup_message))

    # Parse the arguements
    parser = MyArgParser()
    parser.add_argument('-method', '--method', type=str, metavar= 'method_name', action='store',
                        help='which method you want to use',
                        choices={'word2vec','swordnet','verbnet','callnet','assignnet','index','rank','idf','kg','fca'},
                        required=True)
    parser.add_argument('-input', '--input', type=str, metavar= 'input_dir', action='store',
                        help='which directory is containing corpus (e.g. eclipse path)',
                        required=False)
    parser.add_argument('-output', '--output', type=str, metavar= 'output_dir', action='store',
                        help='which directory is output',
                        required=False)
    args = parser.parse_args()

    # Call Jinqui2014 to prepare knowledge base of word/phrase pairs from comment sentences and code identifiers
    if args.method == "swordnet":
        swordnet = SWordNet()
        swordnet.data_path = args.input
        if args.input.split('/')[-1]!='': swordnet.project_name = args.input.split('/')[-1]
        else: swordnet.project_name = args.input.split('/')[-2]
        swordnet.output_path = args.output
        swordnet.find_pairs()

    # Call Matthew2013 to prepare knowledge base of verbs from leading comments and method names
    if args.method == "verbnet":
        verbnet = VerbNet()
        verbnet.data_path = args.input
        if args.input.split('/')[-1]!='': verbnet.project_name = args.input.split('/')[-1]
        else: verbnet.project_name = args.input.split('/')[-2]
        verbnet.output_path = args.output
        verbnet.find_pairs()

if __name__ == "__main__":
    main(sys.argv)
