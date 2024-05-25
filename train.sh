#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd
#$ -l USE_SSH=1


source /etc/profile.d/modules.sh
module load singularitypro
export EXP_PATH=$(pwd)
singularity exec --nv llm-train.sif -c "cd $EXP_PATH && accelerate launch llm-ddp-train.py"