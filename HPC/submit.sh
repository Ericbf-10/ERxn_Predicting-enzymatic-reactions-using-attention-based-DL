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
module load cuda/11.1
module load python3/3.9.11
module load pandas/1.4.1-python-3.7.11
module load numpy/1.22.3-python-3.9.11-openblas-0.3.19

#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#pip3 install torchtext
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torchtext==0.10.1
pip3 install --upgrade pip
pip3 install  networkx
pip3 install --user matplotlib
pip3 install tqdm
pip3 install biopython
pip3 install requests
pip3 install -U scikit-learn


#python3 /zhome/4c/8/164840/projects/ERxn/scripts/12_hyper_param_optimization.py
#python3 /zhome/4c/8/164840/projects/ERxn/scripts/10_protein_autoencoder.py

CUDA_LAUNCH_BLOCKING=1 python3 /zhome/4c/8/164840/projects/ERxn/scripts/13_mol_transformer.py
#python3 /zhome/4c/8/164840/projects/ERxn/scripts/11_protein_encoder.py
