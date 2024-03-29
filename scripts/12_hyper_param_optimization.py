#!/usr/bin/env python3
# coding: utf-8

## IMPORT LIBRARIES ##
import os
import subprocess
import time

## MAIN CODE ##

# Define a list with all hyperparameter variables used and string to do bash call
hyper_param_list = []
hyper_param_string = []

# Define ranges that make sense for each hyperparameter
#LEARNING_RATE = [0.0001, 0.001, 0.01, 0.1] # Default=0.000001
#hyper_param_list.append(LEARNING_RATE)
#hyper_param_string.append("-lr")

#WEIGHT_DECAY = [0.00001, 0.0001, 0.001, 0.01] # Default=0.0001
#hyper_param_list.append(WEIGHT_DECAY)
#hyper_param_string.append("-wd")

#NUM_EPOCHS = [10000, 5000, 1000, 500, 100] # Default=1000
#hyper_param_list.append(NUM_EPOCHS)
#hyper_param_string.append("-epoch")

#PATIENCE = [30, 20, 10, 5] # Default=10
#hyper_param_list.append(PATIENCE)
#hyper_param_string.append("-pati")

#BATCH_SIZE = [256, 180, 100, 30] # Default=100
#hyper_param_list.append(BATCH_SIZE)
#hyper_param_string.append("-bs")

#MOMENTUM = [0.9, 0.8, 0.7, 0.6] # Default=0.9
#hyper_param_list.append(MOMENTUM)
#hyper_param_string.append("-m")

PATCH_LENGTH = [400, 200, 100] # Default=400
hyper_param_list.append(PATCH_LENGTH)
hyper_param_string.append("-plen")

DEPTH = [30, 24, 12] # Default=12
hyper_param_list.append(DEPTH)
hyper_param_string.append("-depth")

N_HEADS = [24, 16, 12] # Default=12; All of them divisors of 768 (default EMBED_DIM)
hyper_param_list.append(N_HEADS)
hyper_param_string.append("-heads")

EMBED_DIM = [504, 768, 1075] # Default=768; All of them are divisible by 12 (default N_HEADS)
hyper_param_list.append(EMBED_DIM)
hyper_param_string.append("-embed")

#MLP_RATIO = [8.0, 6.0, 4.0, 2.0] # Default=4.0
#hyper_param_list.append(MLP_RATIO)
#hyper_param_string.append("-mlp")

#P_DROP = [0.0, 0.1, 0.2, 0.4] # Default=0.1
#hyper_param_list.append(P_DROP)
#hyper_param_string.append("-p")

#ATTN_P = [0.0, 0.1, 0.2, 0.4] # Default=0.1
#hyper_param_list.append(ATTN_P)
#hyper_param_string.append("-attnp")

#QKV_BIAS = [True, False] # Default=True
#hyper_param_list.append(QKV_BIAS)
#hyper_param_string.append("-qkvbias")

#PIN_MEMORY = [False, True] # Default=False
#hyper_param_list.append(PIN_MEMORY)
#hyper_param_string.append("-pin")

script_path = os.path.dirname(__file__)
results_path = os.path.join(script_path, "../results/hyper_param_benchmark")
hpc_path = os.path.join(script_path, '../HPC/hyperparam.sh')

# Submit a job in the HPC queue for each value of each hyperparameter
for i in range(len(hyper_param_list)):
    for value in hyper_param_list[i]:
        flag = True
        out_file = "summary" + hyper_param_string[i] + "=" + str(value)
        job1 = subprocess.Popen(["bsub <", hpc_path, hyper_param_string[i], str(value), "-fout", out_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        #(out, err) = job1.communicate() # Only for debugging purposes
        while flag:
            job2 = subprocess.Popen(["ls", results_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            (out, err) = job2.communicate()
            outfile_list = out.split('\n')
            outfile_list.pop()
            out_file = out_file + ".txt"
            if out_file not in outfile_list:
                time.sleep(180)
            else:
                flag = False
