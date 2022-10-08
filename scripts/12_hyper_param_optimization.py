#!/usr/bin/env python3
# coding: utf-8

## IMPORT LIBRARIES ##
import os
import subprocess

## FUNCTIONS ##

## MAIN CODE ##

# Define paths
script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
point_cloud_path = os.path.join(data_dir, 'point_cloud_dataset')
results_dir = os.path.join(script_path, '../results')

# Define ranges that make sense for each hyperparameter
LEARNING_RATE = [0.0001, 0.001, 0.01, 0.1] # Default=0.001
WEIGHT_DECAY = [0.00001, 0.0001, 0.001, 0.01] # Default=0.0001
NUM_EPOCHS = [10000, 5000, 1000, 500, 100] # Default=1000
PATIENCE = [100, 50, 10, 5] # Default=10
BATCH_SIZE = [256, 180, 100, 30] # Default=100
MOMENTUM = [0.9, 0.8, 0.7, 0.6] # Default=0.9
PATCH_LENGTH = [600, 500, 400, 200] # Default=400
EMBED_DIM = args.EMBED_DIM # Default=768
DEPTH = args.DEPTH # Default=12
N_HEADS = args.N_HEADS # Default=12
MLP_RATIO = args.MLP_RATIO # Default=4.0
QKV_BIAS = args.QKV_BIAS # Default=True
P_DROP = args.P_DROP # Default=0.
ATTN_P = args.ATTN_P # Default=0.
# MISSING PIN_MEMORY
# MISSING RESUME_TRAINING

# Run the 11_protein_encoder.py for each value of 1 hyperparameter using queuing system
# Write results to csv file
# Repeat for every hyperparameter