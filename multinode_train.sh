#!/bin/bash

#$ -l rt_F=4
#$ -l h_rt=4:00:00
#$ -j y
#$ -cwd
#$ -l USE_SSH=1
#$ -o multi-node.out
#$ -e multi-node.err

source /etc/profile.d/modules.sh
module load singularitypro hpcx-mt/2.12
cat ${SGE_JOB_HOSTLIST} | awk '{print $0, "slots=4"}' > hostfile
export MASTER_HOST=$(cat ${SGE_JOB_HOSTLIST} | head -n 1)
export MASTER_IP=$(dig +short $MASTER_HOST)
export HOST0=$(cat ${SGE_JOB_HOSTLIST} | head -n 1 | tail -n 1)
export HOST1=$(cat ${SGE_JOB_HOSTLIST} | head -n 2 | tail -n 1)
export HOST2=$(cat ${SGE_JOB_HOSTLIST} | head -n 3 | tail -n 1)
export HOST3=$(cat ${SGE_JOB_HOSTLIST} | head -n 4 | head -n 1)
export MASTER_PORT=29500
export EXP_PATH=$(pwd)


ssh -p 2222 $HOST0 "cd $EXP_PATH && source /etc/profile.d/modules.sh && module load singularitypro && singularity exec --nv --bind $EXP_PATH:$EXP_PATH llm-train.sif  "bash multi-command.sh $EXP_PATH $MASTER_IP $MASTER_PORT 0"" &
ssh -p 2222 $HOST1 "cd $EXP_PATH && source /etc/profile.d/modules.sh && module load singularitypro && singularity exec --nv --bind $EXP_PATH:$EXP_PATH llm-train.sif "bash multi-command.sh $EXP_PATH $MASTER_IP $MASTER_PORT 1"" &
ssh -p 2222 $HOST2 "cd $EXP_PATH && source /etc/profile.d/modules.sh && module load singularitypro && singularity exec --nv --bind $EXP_PATH:$EXP_PATH llm-train.sif "bash multi-command.sh $EXP_PATH $MASTER_IP $MASTER_PORT 2"" &
ssh -p 2222 $HOST3 "cd $EXP_PATH && source /etc/profile.d/modules.sh && module load singularitypro && singularity exec --nv --bind $EXP_PATH:$EXP_PATH llm-train.sif "bash multi-command.sh $EXP_PATH $MASTER_IP $MASTER_PORT 3""