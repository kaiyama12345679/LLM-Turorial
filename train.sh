#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd
#$ -l USE_SSH=1
#$ -o train.out
#$ -e train.err


source /etc/profile.d/modules.sh
module load singularitypro
export EXP_PATH=$(pwd)
singularity exec --nv --bind $EXP_PATH:$EXP_PATH llm-train.sif bash command.sh
