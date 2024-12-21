
#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J cli_project
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- set to maximum 24 hours for GPU-queues
#BSUB -W 24:00
### -- request 20GB of system memory --
#BSUB -R "rusage[mem=20GB]"
### -- set the email address (uncomment and replace with your email) --
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o cli_project_%J.out
#BSUB -e cli_project_%J.err
# -- end of LSF options --

# Display GPU information
nvidia-smi

# Load the required CUDA module
module load cuda/11.8

# Activate your Python virtual environment
source ~/dl_pro/bin/activate

# Run your Python script with specified arguments
python ~/dl_pro/data/data/Structured_SSM/cli_revised.py \
    --output_path=/zhome/11/b/204396/dl_pro/data/data/transformer \
    --base_path=/zhome/11/b/204396/dl_pro/data/data/Structured_SSM/P12data \
    --epochs=100 \
    --batch_size=16 \
    --model_type=transformer \
    --dropout=0.2 \
    --attn_dropout=0.1 \
    --layers=3 \
    --heads=1 \
    --pooling=max \
    --lr=0.0001

