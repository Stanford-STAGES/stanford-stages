#!/usr/bin/env bash
# This script should be run from the root of the stanford-stages directory
conda env update --file environment.yml
conda activate stanford-stages
pip install -r requirements.txt

wget http://www.informaton.org/narco/ml/ac.zip
unzip ac.zip -d ./ml/

wget http://www.informaton.org/narco/ml/gp.zip
unzip gp.zip -d ./ml/
