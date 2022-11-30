#!/bin/sh

### -- set the job Name --
#BSUB -J train
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- specify queue -- voltash cabgpu gpuv100
#BSUB -q cabgpu
### -- set walltime limit: hh:mm --
#BSUB -W 200:00
### -- Select the resources: 1 gpu in exclusive process mode --:mode=exclusive_process
#BSUB -gpu "num=1:mode=exclusive_process"
## --- select a GPU with 32gb----
#BSUB -R "select[gpu40gb]"
### -- specify that we need 3GB of memory per core/slot --
#BSUB -R "rusage[mem=64GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train.out
#BSUB -e train.err

# here follow the commands you want to execute

# submit with bsub < submit.sh
>train.out
>train.err

cd ~/projects/ERxn/scripts
module load cuda/11.6
module load python3/3.10.7
module load pandas/1.4.4-python-3.10.7

#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#pip3 install torchtext
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchtext==0.6.0
pip install --upgrade pip
pip install  networkx
pip install --user matplotlib
pip install tqdm
pip install biopython
pip install requests
pip install -U scikit-learn


#python3 /zhome/4c/8/164840/projects/ERxn/scripts/12_hyper_param_optimization.py
#python3 /zhome/4c/8/164840/projects/ERxn/scripts/10_protein_autoencoder.py

#python3 /zhome/4c/8/164840/projects/ERxn/scripts/13_mol_transformer.py
python3 /zhome/4c/8/164840/projects/ERxn/scripts/11_protein_encoder.py -optim adam -plen 100
