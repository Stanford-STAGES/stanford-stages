#!/usr/bin/env bash
env_name=${1:-stanford-stages}

echo "This script should be run from the root of the stanford-stages directory"

conda env create --name $env_name --file environment.yml
# make conda activate runnable in non-interactive shell
eval "$(conda shell.bash hook)" 
conda activate $env_name
pip install -r requirements.txt

wget http://www.informaton.org/narco/ml/ac.zip
unzip ac.zip -d ./ml/

wget http://www.informaton.org/narco/ml/gp.zip
unzip gp.zip -d ./ml/
