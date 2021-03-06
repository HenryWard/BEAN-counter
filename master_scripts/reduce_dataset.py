#!/usr/bin/env python

#################################################################
######  Copyright: Regents of the University of Minnesota  ######
#################################################################

# This script takes in a dataset and a corresponding sample table with
# a reduced number of rows. This script will reduce the dataset so that
# it contains only the samples corresponding to the rows in the
# dataset. It also exports a sample table that perfectly corresponds
# to the matrix (any extra rows in the sample table have been removed).
import argparse
import numpy as np, pandas as pd
import networkx as nx
import os, sys
import gzip
import cPickle

def read_sample_table(tab_filename):

    # Read everything in as a string, to prevent vexing
    # number interpretation problems! Methods further down
    # can coerce to different types.
    tab = pd.read_table(tab_filename, dtype = 'S')
    return tab

def load_dataset(data_filename):

    f = gzip.open(data_filename, 'rb')
    barcodes, conditions, matrix = cPickle.load(f)
    dataset = [np.array(barcodes), np.array(conditions), matrix]
    f.close()
    return dataset

def dump_dataset(dataset, filename):

    f = gzip.open(filename, 'wb')
    cPickle.dump(dataset, f)
    f.close()

def filter_sample_table(sample_table, final_conditions):

    all_conditions = np.array([tuple(row[['screen_name', 'expt_id']]) for i,row in sample_table.iterrows()])
    rows_to_keep = [i for i, val in enumerate(all_conditions) if a_is_row_in_b(val, final_conditions)]
    #print final_conditions
    #print all_conditions
    #print rows_to_keep

    return sample_table.iloc[rows_to_keep]

def filter_matrix(sample_table, matrix_conditions, matrix):

    reduced_conditions = np.array([tuple(row[['screen_name', 'expt_id']]) for i,row in sample_table.iterrows()])
    matrix_conds_to_keep = np.array([i for i, val in enumerate(matrix_conditions) if a_is_row_in_b(val, reduced_conditions)])
    #print final_conditions
    #print all_conditions
    #print rows_to_keep

    return matrix_conditions[matrix_conds_to_keep], matrix[:, matrix_conds_to_keep]

def a_is_row_in_b(a, b):

    return np.any(np.all(a == b, axis = 1))


def main(dataset, sample_table, output_folder, verbosity):

    strains, conditions, matrix = dataset

    # First, filter the sample table to make sure it doesn't have extra conditions
    sample_table = filter_sample_table(sample_table, conditions)

    # Then, reduce the matrix and condition array
    reduced_conditions, reduced_matrix = filter_matrix(sample_table, conditions, matrix)

    if verbosity >= 2:
        print sample_table
        print reduced_conditions.shape
        print reduced_matrix.shape

    dataset_filename = os.path.join(output_folder, 'reduced_dataset.dump.gz')
    table_filename = os.path.join(output_folder, 'reduced_sample_table.txt')

    dump_dataset([strains, reduced_conditions, reduced_matrix], dataset_filename)
    sample_table.to_csv(table_filename, sep = '\t', header = True, index = False)

# If this is run from the command line (which is the intention), then run the rest of this script!
if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', help = 'The dataset to be reduced.')
    parser.add_argument('sample_table', help = 'The sample table that contains fewer conditions than are present in the dataset.')
    parser.add_argument('output_folder', help = 'The folder to which the resulting reduced matrix and sample table are written')    
    parser.add_argument('-v', '--verbosity', help = 'The level of verbosity printed to stdout. Ranges from 0 to 3, 1 is default.')

    args = parser.parse_args()

    # Take care of verbosity right away
    if args.verbosity is None:
        verbosity = 1
    elif args.verbosity.isdigit():
        verbosity = int(args.verbosity)
    else:
        verbosity = 1

    # Get the data ready to rumble!
    dataset_file = os.path.abspath(args.dataset_file)
    assert os.path.isfile(dataset_file), "Dataset file does not exist."
    dataset = load_dataset(args.dataset_file)

    if verbosity >= 2:
        print dataset

    assert os.path.isfile(args.sample_table), "Sample table file does not exist."
    sample_table = read_sample_table(args.sample_table)

    if verbosity >= 2:
        print sample_table
        print dataset

    output_folder = args.output_folder
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    main(dataset, sample_table, output_folder, verbosity)

