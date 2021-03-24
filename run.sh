#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -t 3-00:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -c 12
#SBACTH -g 2
#SBATCH --mail-type=END,FAIL

#source venv/bin/activate
#module load python/3.8.3

## Small Dataset
#python -m qroute --dataset small --train

# Random tests
FILE="test/test_results/random_results.txt"
echo "Starting Random Testing"

for i in 30 50 70 90 110 130 150
do
  echo ""
  echo "Starting Routing on $i"
  echo ""
  python -m qroute --dataset random --search 200 --gates $i --iterations 10
  mkdir test/compiled_circuits/random_$i/
  mv test/test_results/*.json test/compiled_circuits/random_$i/
done
