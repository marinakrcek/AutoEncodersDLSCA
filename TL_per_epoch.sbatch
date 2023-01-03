#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=attackNN_search
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:40:00

# 500MB memory per core
# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8411MB
#SBATCH --gres=gpu
#SBATCH --mail-type=END,FAIL

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/11.2 cudnn/11.2-8.1.1.33
module list

cd /home/nfs/mkrcek
source venv/bin/activate
cd /home/nfs/mkrcek/AutoEncodersDLSCA

srun python attack_in_latent_transfer_learning.py $1 $2 $3 $4 $5 $6
exit 0


