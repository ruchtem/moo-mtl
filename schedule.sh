#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem 65000 # memory pool for each core (4GB)
#SBATCH -t 1-23:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o tmp/log/%x.%j.out # stdout and stderr
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)


source $HOME/dev/venvs/base/bin/activate

IFS='-'; arrIN=($SLURM_JOB_NAME); unset IFS;


COMMAND="python -u multi_objective/main.py --method ${arrIN[0]} --config configs/${arrIN[1]}.yaml --tag ${arrIN[2]}";
echo "Workingdir: $PWD";
echo "Started at $(date) on host $SLURMD_NODENAME";
echo "Executing $COMMAND";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

eval $COMMAND;

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
