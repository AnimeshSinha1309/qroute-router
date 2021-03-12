#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -t 3-00:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -c 12
#SBACTH -g 2
#SBATCH --mail-type=END,FAIL

source venv/bin/activate
module load python/3.8.3
python -m qroute --dataset small --train

