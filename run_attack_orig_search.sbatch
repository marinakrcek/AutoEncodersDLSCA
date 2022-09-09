#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=attackNN_search
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:30:00

# 500MB memory per core
# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1500MB
#SBATCH --gres=gpu
#SBATCH --mail-type=END

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.6.0.64
module list

cd /home/nfs/mkrcek
source venv/bin/activate
cd /home/nfs/mkrcek/AutoEncodersDLSCA

srun python attack_orig_space.py $1 $2 $3
exit 0


