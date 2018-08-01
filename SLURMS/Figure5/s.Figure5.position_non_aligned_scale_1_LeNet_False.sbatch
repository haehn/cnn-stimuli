#!/bin/bash
#SBATCH -p cox
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --array=1-4
#SBATCH --mem=8000
#SBATCH -t 3-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ian.svetkey@gmail.com
#SBATCH -o /n/home05/isvetkey/cnn-stimuli/SLURMS/Figure5/LOGS/%A_%a.out
#SBATCH -e /n/home05/isvetkey/cnn-stimuli/SLURMS/Figure5/LOGS/%A_%a.err

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

source new-modules.sh

module load Anaconda/5.0.1-fasrc01

cd /n/home05/isvetkey/cnn-stimuli/

source activate CP

python run_regression_isvetkey.py s.Figure5.position_non_aligned_scale 1 LeNet False $SLURM_ARRAY_TASK_ID 

# end
exit 0;
