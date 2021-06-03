#### START SLURM CONFIGURATION ####

#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nome.cognome@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./test/test.log
#SBATCH --gres=gpu:1

#### END SLURM CONFIGURATION ####

#### START USER SCRIPT ####

source ./v_env1/bin/activate # activates the Python virtual environment, installed under 'venv' folder

export TORCH_HOME="models/pretrainedmodels"

bash ./test.sh

#### END USER SCRIPT ####
