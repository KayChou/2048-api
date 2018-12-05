#!/bin/bash	
#SBATCH	-J	2048CNN
#SBATCH	-p	gpu	
#SBATCH	-N	1	
#SBATCH	--output=log.%j.out	
#SBATCH	--error=log.%j.err	
#SBATCH	-t	30:00:00	
#SBATCH	--gres=gpu:1
module	load	anaconda3/5.3.0	
source	activate python35
python main.py
