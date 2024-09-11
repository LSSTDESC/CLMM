import os, sys
import numpy as np
number_of_splits = 20
for which_split in np.arange(number_of_splits):
    lines_base = [
        '#!/bin/sh',
        '# SBATCH options:',
        '#SBATCH --job-name=extract_cosmodc2_redmapper    # Job name',
        f'#SBATCH --output=./log/{which_split}_nsplits_{number_of_splits}.log',
        '#SBATCH --partition=htc               # Partition choice',
        '#SBATCH --ntasks=1                    # Run a single task (by default tasks == CPU)',
        '#SBATCH --mem=8000                    # Memory in MB per default',
        '#SBATCH --time=0-6:00:00             # 7 days by default on htc partition',
        'source /pbs/home/c/cpayerne/setup_mydesc.sh']
    cmd =   f'python run_extract_sources_and_compute_ind_profile_cosmoDC2_per_split.py --which_split {which_split} --number_of_splits {number_of_splits} '
    lines = lines_base + [cmd]
    name_job = f'job_which_split={which_split}_number_of_splits={number_of_splits}.job'
    with open(name_job, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    #os.system(f'sbatch {name_job}')
    os.remove(name_job)
