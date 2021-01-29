#!/bin/bash

sbatch -J cosmos_ln-adult-1 schedule.sh
sbatch -J cosmos_ln-adult-2 schedule.sh
sbatch -J cosmos_ln-adult-3 schedule.sh
sbatch -J cosmos_ln-adult-4 schedule.sh
sbatch -J cosmos_ln-adult-42 schedule.sh

sbatch -J cosmos_ln-compas-1 schedule.sh
sbatch -J cosmos_ln-compas-2 schedule.sh
sbatch -J cosmos_ln-compas-3 schedule.sh
sbatch -J cosmos_ln-compas-4 schedule.sh
sbatch -J cosmos_ln-compas-42 schedule.sh

sbatch -J cosmos_ln-credit-1 schedule.sh
sbatch -J cosmos_ln-credit-2 schedule.sh
sbatch -J cosmos_ln-credit-3 schedule.sh
sbatch -J cosmos_ln-credit-4 schedule.sh
sbatch -J cosmos_ln-credit-42 schedule.sh

sbatch -J cosmos_ln-mm-1 schedule.sh
sbatch -J cosmos_ln-mm-2 schedule.sh
sbatch -J cosmos_ln-mm-3 schedule.sh
sbatch -J cosmos_ln-mm-4 schedule.sh
sbatch -J cosmos_ln-mm-42 schedule.sh

sbatch -J cosmos_ln-fm-1 schedule.sh
sbatch -J cosmos_ln-fm-2 schedule.sh
sbatch -J cosmos_ln-fm-3 schedule.sh
sbatch -J cosmos_ln-fm-4 schedule.sh
sbatch -J cosmos_ln-fm-42 schedule.sh

sbatch -J cosmos_ln-mfm-1 schedule.sh
sbatch -J cosmos_ln-mfm-2 schedule.sh
sbatch -J cosmos_ln-mfm-3 schedule.sh
sbatch -J cosmos_ln-mfm-4 schedule.sh
sbatch -J cosmos_ln-mfm-42 schedule.sh



# sbatch -J hyper_ln-adult schedule.sh
# sbatch -J hyper_ln-compas schedule.sh
# sbatch -J hyper_ln-credit schedule.sh
# sbatch -J hyper_ln-mm schedule.sh
# sbatch -J hyper_ln-fm schedule.sh
# sbatch -J hyper_ln-mfm schedule.sh

# sbatch -J hyper_epo-adult schedule.sh
# sbatch -J hyper_epo-compas schedule.sh
# sbatch -J hyper_epo-credit schedule.sh
# sbatch -J hyper_epo-mm schedule.sh
# sbatch -J hyper_epo-fm schedule.sh
# sbatch -J hyper_epo-mfm schedule.sh

# sbatch -J single_task-adult schedule.sh
# sbatch -J single_task-compas schedule.sh
# sbatch -J single_task-credit schedule.sh
# sbatch -J single_task-mm schedule.sh
# sbatch -J single_task-fm schedule.sh
# sbatch -J single_task-mfm schedule.sh

# sbatch -J pmtl-adult schedule.sh
# sbatch -J pmtl-compas schedule.sh
# sbatch -J pmtl-credit schedule.sh
# sbatch -J pmtl-mm schedule.sh
# sbatch -J pmtl-fm schedule.sh
# sbatch -J pmtl-mfm schedule.sh
