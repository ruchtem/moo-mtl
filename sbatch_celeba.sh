#!/bin/bash


# cosmos
# sbatch -J cosmos_ln-celeba-1-0 schedule_celeba.sh

# uniform
# sbatch -J uniform-celeba-1-0 schedule_celeba.sh

# mgda
# sbatch -J mgda-celeba-1-0 schedule_celeba.sh


# single task in parallel
# sbatch -J single_task-celeba-1-16 schedule_celeba.sh    # 3 tasks
# sbatch -J single_task-celeba-1-22 schedule_celeba.sh
# sbatch -J single_task-celeba-1-24 schedule_celeba.sh    

# sbatch -J single_task-celeba-1-16 schedule_celeba.sh    # 4th task
# sbatch -J single_task-celeba-1-22 schedule_celeba.sh
# sbatch -J single_task-celeba-1-24 schedule_celeba.sh    
# sbatch -J single_task-celeba-1-26 schedule_celeba.sh    

# sbatch -J single_task-celeba-1-8 schedule_celeba.sh     # 10 random tasks
# sbatch -J single_task-celeba-1-36 schedule_celeba.sh
# sbatch -J single_task-celeba-1-4 schedule_celeba.sh
# sbatch -J single_task-celeba-1-16 schedule_celeba.sh
# sbatch -J single_task-celeba-1-7 schedule_celeba.sh
# sbatch -J single_task-celeba-1-31 schedule_celeba.sh
# sbatch -J single_task-celeba-1-28 schedule_celeba.sh
# sbatch -J single_task-celeba-1-30 schedule_celeba.sh
# sbatch -J single_task-celeba-1-24 schedule_celeba.sh
# sbatch -J single_task-celeba-1-13 schedule_celeba.sh

